import numpy as np
import torch
"""
使用numpy
"""
def use_np():
    # 超参数设置
    N, x_dim, h_dim, y_dim = 100, 1000, 100, 10
    lr = 1e-6
    # 样本初始化
    X = np.random.randn(N, x_dim)
    y = np.random.randn(N, y_dim)
    # 参数矩阵初始化
    W_1 = np.random.randn(x_dim, h_dim)
    W_2 = np.random.randn(h_dim, y_dim)
    for step in range(500):
        # 前向传播
        h = X.dot(W_1)  # (100,100)
        h_relu = np.maximum(0, h)
        y_bar = h_relu.dot(W_2)  # (100,10)
        # 损失函数
        J = np.square(y_bar - y).sum()
        # 反向传播
        grad_y_bar = 2.0 * (y_bar - y)  # (100,10)
        grad_w2 = h_relu.T.dot(grad_y_bar)  # (100,10)
        grad_h_relu = grad_y_bar.dot(W_2.T)  # (100,100)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = X.T.dot(grad_h)  # (1000,100)
        # 参数更新
        W_1 -= lr * grad_w1
        W_2 -= lr * grad_w2
    #     print(step, J)
def use_non_nn():
    dtype = torch.float
    device = torch.device("cpu")
    # 超参数初始化
    N, x_dim, h_dim, y_dim = 64, 1000, 100, 10
    # 训练数据和参数初始化
    X = torch.randn(N, x_dim, device=device, dtype=dtype)
    Y = torch.randn(N, y_dim, device=device, dtype=dtype)
    w1 = torch.randn(x_dim, h_dim, device=device, dtype=dtype)
    w2 = torch.randn(h_dim, y_dim, device=device, dtype=dtype)
    lr = 1e-6
    for step in range(500):
        # 前向暂停
        h = X.mm(w1)  # (N,h_dim)不明白查官方文档，不行百度
        h_relu = h.clamp(min=0)  # (N,h_dim)
        y_bar = h_relu.mm(w2)  # (N,y_dim)
        # 定义损失函数
        loss = (y_bar - Y).pow(2).sum().item()
        #     print(step, loss)
        # 反向暂停
        grad_y_bar = 2 * (y_bar - Y)  # (N,y_dim)
        grad_w2 = h_relu.t().mm(grad_y_bar)  # (h_dim,y_dim)
        grad_h_relu = grad_y_bar.mm(w2.t())  # (N,h_dim)
        grad_h = grad_h_relu.clone()  # numpy里用copy()
        grad_h[h < 0] = 0
        grad_w1 = X.t().mm(grad_h)
        # 参数更新
        w1 -= lr * grad_w1
        w2 -= lr * grad_w2


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_bar = self.linear2(h_relu)
        return y_bar


def use_NN():
    N, x_dim, h_dim, y_dim = 64, 1000, 100, 10
    X = torch.randn(N, x_dim)
    Y = torch.randn(N, y_dim)
    model = TwoLayerNet(x_dim, h_dim, y_dim)
    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for step in range(500):
        y_bar = model(X)
        loss = criterion(y_bar, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(step, loss.item())
if __name__ == '__main__':
    use_NN()