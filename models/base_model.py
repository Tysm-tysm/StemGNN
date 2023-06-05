import torch
import torch.nn as nn
import torch.nn.functional as F


# 门控单元paper: Language Modeling with Gated Convolutional Networks
# 作用:1序列深度建模; 2.减轻梯度弥散，加速收敛
class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class StockBlockLayer(nn.Module):
    def __init__(self, time_step, unit, multi_layer, stack_cnt=0):
        super(StockBlockLayer, self).__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt
        self.multi = multi_layer
        self.weight = nn.Parameter(
            torch.Tensor(1, 3 + 1, 1, self.time_step * self.multi,
                         self.multi * self.time_step))  # [K+1, 1, in_c, out_c]
        nn.init.xavier_normal_(self.weight)
        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)
        if self.stack_cnt == 0:
            self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)
        # 数据原特征表达
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)
        self.relu = nn.ReLU()
        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi
        for i in range(3):
            if i == 0:
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
            elif i == 1:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
            else:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))

    def spe_seq_cell(self, input):
        # input shape: torch.Size([32, 4, 1, 140, 12])
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, -1, node_cnt, time_step)

        # fft: 快速离散傅里叶变换, rfft: 去除那些共扼对称的值，减小存储
        ffted = torch.rfft(input, 1, onesided=False)
        # ffted shape: torch.Size([32, 4, 140, 12, 2])
        real = ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        img = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)

        # GLU
        for i in range(3):
            real = self.GLUs[i * 2](real)
            img = self.GLUs[2 * i + 1](img)
        # real shape: torch.Size([32，140，240])

        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        # real shape: torch.Size([32, 4, 140, 60])

        time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
        #  torch.Size([32, 4, 140, 60, 2])

        # IDFT
        iffted = torch.irfft(time_step_as_inner, 1, onesided=False)
        return iffted

    def forward(self, x, mul_L):

        # x shape: torch.Size([32, 1, 140, 12])
        # mul L shape: torch.Size([4, 140, 140])
        mul_L = mul_L.unsqueeze(1)  # torch.Size([4, 1, 140, 140])
        x = x.unsqueeze(1)

        # learning latent representations of multiple time-series in the spectral domain
        # torch.matmul 支持广播机制
        gfted = torch.matmul(mul_L, x)
        # gfted shape: torch.Size([32, 4, 1, 10, 12])

        # captures the repeated patterns in the periodic data
        # or the auto-correlation features among different timestamps
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)
        # gconv_input shape: torch.Size([32, 4, 1, 140, 60])

        # GConV + IGFT
        # weight: torch.Size([1, 3 + 1, 1, self.time_step * self.multi, self.multi * self.time_step])
        # weight :[1, 4, 1, 60, 60]
        igfted = torch.matmul(gconv_input, self.weight)
        # igfted shape : torch.Size([32, 4, 1, 140,  601)

        igfted = torch.sum(igfted, dim=1)
        # igfted shape : torch.Size([32, 1, 140,  601)

        # ----------------- Forecast
        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        forecast = self.forecast_result(forecast_source)
        # forecast_source shape and forecast shape: torch.Size([32, 140, 60]) torch.Size([32, 140, 12])

        # ----------------- Backcast
        if self.stack_cnt == 0:
            #  x shape: torch.Size([32, 1, 1, 140, 12])
            backcast_short = self.backcast_short_cut(x).squeeze(1)
            #  backcast_short shape: torch.Size([32, 1, 140, 12])
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
        else:
            backcast_source = None
        # backcast_source shape: torch.Size([32, 1, 140, 12]) or None
        return forecast, backcast_source


class Model(nn.Module):
    def __init__(self, units, stack_cnt, time_step, multi_layer, horizon=1, dropout_rate=0.5, leaky_rate=0.2,
                 device='cuda:0'):
        super(Model, self).__init__()
        self.unit = units
        # StemGNN Block
        self.stack_cnt = stack_cnt
        self.alpha = leaky_rate
        # windows size, is the length of the input sequence
        self.time_step = time_step
        # H predictions in the future, after time t
        self.horizon = horizon
        # self-attention
        #  对GRU的最后一个 隐藏状态R 使用self-attention的方式计算邻接矩阵
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)  # 采用xavier_uniform_这种初始化方法
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)  # 采用xavier_uniform_这种初始化方法
        # nn.GRU parameters: input size, hidden size, num layers=1
        # args.window _size = self.time step
        self.GRU = nn.GRU(self.time_step, self.unit)
        self.multi_layer = multi_layer
        self.stock_block = nn.ModuleList()  # ModuleList block参数将计算在主模型中, 没有顺序要求
        self.stock_block.extend(
            [StockBlockLayer(self.time_step, self.unit, self.multi_layer, stack_cnt=i) for i in range(self.stack_cnt)])
        self.fc = nn.Sequential(
            nn.Linear(int(self.time_step), int(self.time_step)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step), self.horizon),
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.to(device)

    def get_laplacian(self, graph, normalize):
        """
        return the laplacian of the graph.
        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian

    def latent_correlation_layer(self, x):
        # is there has a question ? self.windows
        # x: (batch, sequence, features)
        # GRU default input: (sequence, batch, features), default batch_first=False
        # However， input is (features, batch， sequence) here.
        # torch.Size([140, 32,12]) sequence(window size)=12, but it equals 140 here
        # print("_---GRU input shape: "， x.permute(2, , 1).contiguous().shape)
        input, _ = self.GRU(x.permute(2, 0, 1).contiguous())
        # print("_---GRU output shape:"， input.shape)
        # last state output shape of GRU in doc: (D * num layers, batch, output _size(self.unit))
        # However, (sequence, batch, D*Hout(output_size)) when batch first=False here ???
        # Only all output features is senseful in this situation
        # torch.Size([140，32，140])
        input = input.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(input)  # torch.Size([32, 140, 140])
        attention = torch.mean(attention, dim=0)  # torch.Size([140, 140])
        degree = torch.sum(attention, dim=1)  # torch.Size([140])
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)  # 对称
        degree_l = torch.diag(degree)  # 返回一个以degree为对角线元素的2D矩阵，torch.size([140，140])
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))  # torch.Size([140, 140])
        mul_L = self.cheb_polynomial(laplacian)  # 多阶拉普拉斯矩阵, torch.Size([140, 140])
        return mul_L, attention

    def self_graph_attention(self, input):
        # input shape here: (batch, sequence, output_size)
        input = input.permute(0, 2, 1).contiguous()
        #  after trans:(batch, output_size, sequence)
        #  this is  why input == output ?
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        #  key shape: torch.Size([32， 140， 1])
        query = torch.matmul(input, self.weight_query)
        #  torch.repeat 当参数有三个时: (通道数的重复倍数，行的重复倍数，列的重复倍数)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        # data shape : torch.size([32，143 *140,1])
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    def forward(self, x):
        # part 1
        mul_L, attention = self.latent_correlation_layer(x)
        # X: (batch, sequence, features) == > X: (batch, 1， features， sequence)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()

        # part 2
        result = []
        for stack_i in range(self.stack_cnt):  # stack_i帮助判断是第几个block，因为两个不同的block有一些区别。
            # X shape : torch.Size([32, 1, 140, 12])
            # mul_L shape : torch.Size([4, 140, 140])
            forecast, X = self.stock_block[stack_i](X, mul_L)
            # output X : backcast = X - X hat ==> torch.Size([32，1，140，12])
            result.append(forecast)

        forecast = result[0] + result[1]  # torch.Size([32, 140, 12])

        forecast = self.fc(forecast)

        if forecast.size()[-1] == 1:
            # forecast shape: torch.Size([32, 140, 1]) ==> torch.Size([32, 1, 140])
            return forecast.unsqueeze(1).squeeze(-1), attention
        else:
            # forecast shape: torch.Size([32, 140, 3]) ==> torch.Size([32, 3, 140])
            return forecast.permute(0, 2, 1).contiguous(), attention
