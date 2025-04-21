import math
import random


# 生成训练数据：二进制序列和奇偶标签
def generate_data(sequence_length=3):
    data = []
    for i in range(2 ** sequence_length):
        binary = format(i, f'0{sequence_length}b')
        sequence = [int(bit) for bit in binary]
        label = sum(sequence) % 2  # 1的个数是否为奇数
        data.append((sequence, label))
    return data


# 辅助数学函数（纯Python实现）
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh_derivative(x):
    return 1 - math.tanh(x) ** 2


def vec_add(v1, v2):
    return [x + y for x, y in zip(v1, v2)]


def scalar_vec_mult(s, v):
    return [s * x for x in v]


def mat_vec_mult(matrix, vector):
    return [sum(row[i] * vector[i] for i in range(len(vector))) for row in matrix]


def transpose(matrix):
    return list(map(list, zip(*matrix)))


# 参数初始化（4组门参数）
def initialize_parameters(hidden_size):
    # 输入门参数
    W_xi = [random.uniform(-0.01, 0.01) for _ in range(hidden_size)]
    W_hi = [[random.uniform(-0.01, 0.01) for _ in range(hidden_size)] for _ in range(hidden_size)]
    b_i = [random.uniform(-0.01, 0.01) for _ in range(hidden_size)]

    # 遗忘门参数
    W_xf = [random.uniform(-0.01, 0.01) for _ in range(hidden_size)]
    W_hf = [[random.uniform(-0.01, 0.01) for _ in range(hidden_size)] for _ in range(hidden_size)]
    b_f = [random.uniform(-0.01, 0.01) for _ in range(hidden_size)]

    # 候选值参数
    W_xc = [random.uniform(-0.01, 0.01) for _ in range(hidden_size)]
    W_hc = [[random.uniform(-0.01, 0.01) for _ in range(hidden_size)] for _ in range(hidden_size)]
    b_c = [random.uniform(-0.01, 0.01) for _ in range(hidden_size)]

    # 输出门参数
    W_xo = [random.uniform(-0.01, 0.01) for _ in range(hidden_size)]
    W_ho = [[random.uniform(-0.01, 0.01) for _ in range(hidden_size)] for _ in range(hidden_size)]
    b_o = [random.uniform(-0.01, 0.01) for _ in range(hidden_size)]

    # 输出层参数
    W_hy = [random.uniform(-0.01, 0.01) for _ in range(hidden_size)]
    b_y = random.uniform(-0.01, 0.01)

    return (W_xi, W_hi, b_i,
            W_xf, W_hf, b_f,
            W_xc, W_hc, b_c,
            W_xo, W_ho, b_o,
            W_hy, b_y)


# 前向传播（带详细输出）
def forward_pass(x_sequence, params):
    (W_xi, W_hi, b_i,
     W_xf, W_hf, b_f,
     W_xc, W_hc, b_c,
     W_xo, W_ho, b_o,
     W_hy, b_y) = params

    h_size = len(W_xi)
    h_prev = [0.0] * h_size
    c_prev = [0.0] * h_size

    # 保存所有中间状态
    gates = {
        'i': [], 'f': [], 'c': [], 'o': [],
        'c_state': [], 'h_state': []
    }

    print("\n=== 前向传播开始 ===")
    for t in range(len(x_sequence)):
        x_t = x_sequence[t]

        # 输入门计算（修正点积计算）
        i_t = [
            sigmoid(wx * x_t + sum(wh_j * h_prev_j for wh_j, h_prev_j in zip(wh, h_prev)) + b)
            for wx, wh, b in zip(W_xi, W_hi, b_i)
        ]

        # 遗忘门计算（修正点积计算）
        f_t = [
            sigmoid(wx * x_t + sum(wh_j * h_prev_j for wh_j, h_prev_j in zip(wh, h_prev)) + b)
            for wx, wh, b in zip(W_xf, W_hf, b_f)
        ]

        # 候选值计算（修正点积计算）
        c_hat_t = [
            math.tanh(wx * x_t + sum(wh_j * h_prev_j for wh_j, h_prev_j in zip(wh, h_prev)) + b)
            for wx, wh, b in zip(W_xc, W_hc, b_c)
        ]

        # 更新细胞状态
        c_t = [f * c_prev_i + i * c_hat for f, i, c_prev_i, c_hat in zip(f_t, i_t, c_prev, c_hat_t)]

        # 输出门计算（修正点积计算）
        o_t = [
            sigmoid(wx * x_t + sum(wh_j * h_prev_j for wh_j, h_prev_j in zip(wh, h_prev)) + b)
            for wx, wh, b in zip(W_xo, W_ho, b_o)
        ]

        # 更新隐藏状态
        h_t = [o * math.tanh(c) for o, c in zip(o_t, c_t)]

        # 保存状态
        gates['i'].append(i_t)
        gates['f'].append(f_t)
        gates['c'].append(c_hat_t)
        gates['o'].append(o_t)
        gates['c_state'].append(c_t)
        gates['h_state'].append(h_t)

        # 打印细节
        print(f"\n时间步 {t + 1}/{len(x_sequence)}")
        print(f"输入值: {x_t}")
        print(f"输入门值: {[round(x, 4) for x in i_t]}")
        print(f"遗忘门值: {[round(x, 4) for x in f_t]}")
        print(f"候选值: {[round(x, 4) for x in c_hat_t]}")
        print(f"细胞状态更新: {[round(x, 4) for x in c_t]}")
        print(f"输出门值: {[round(x, 4) for x in o_t]}")
        print(f"新隐藏状态: {[round(x, 4) for x in h_t]}")

        h_prev = h_t
        c_prev = c_t

    # 最终输出计算
    z = sum(w * h for w, h in zip(W_hy, h_prev)) + b_y
    y_pred = sigmoid(z)

    print("\n最终输出计算:")
    print(f"隐藏到输出权重: {[round(x, 4) for x in W_hy]}")
    print(f"最终隐藏状态: {[round(x, 4) for x in h_prev]}")
    print(f"输出偏置: {round(b_y, 4)}")
    print(f"预测值: {round(y_pred, 4)} (实际标签: {label})")

    return y_pred, gates


# 反向传播（带详细输出）(此处需要相应调整点积计算，但原始代码已正确实现)
def backward_pass(x_sequence, y_pred, label, gates, params):
    (W_xi, W_hi, b_i,
     W_xf, W_hf, b_f,
     W_xc, W_hc, b_c,
     W_xo, W_ho, b_o,
     W_hy, b_y) = params

    h_size = len(W_xi)
    T = len(x_sequence)

    # 初始化梯度
    grads = {
        'W_xi': [0.0] * h_size, 'W_hi': [[0.0] * h_size for _ in range(h_size)],
        'b_i': [0.0] * h_size,
        'W_xf': [0.0] * h_size, 'W_hf': [[0.0] * h_size for _ in range(h_size)],
        'b_f': [0.0] * h_size,
        'W_xc': [0.0] * h_size, 'W_hc': [[0.0] * h_size for _ in range(h_size)],
        'b_c': [0.0] * h_size,
        'W_xo': [0.0] * h_size, 'W_ho': [[0.0] * h_size for _ in range(h_size)],
        'b_o': [0.0] * h_size,
        'W_hy': [0.0] * h_size, 'b_y': 0.0
    }

    # 输出层梯度
    delta_y = y_pred - label
    grads['W_hy'] = [delta_y * h for h in gates['h_state'][-1]]
    grads['b_y'] = delta_y

    # 初始化反向传播梯度
    dh_next = [0.0] * h_size
    dc_next = [0.0] * h_size

    print("\n=== 反向传播开始 ===")
    for t in reversed(range(T)):
        # 获取当前时间步的状态
        i_t = gates['i'][t]
        f_t = gates['f'][t]
        o_t = gates['o'][t]
        c_hat_t = gates['c'][t]
        c_t = gates['c_state'][t]
        c_prev = gates['c_state'][t - 1] if t > 0 else [0.0] * h_size
        h_prev = gates['h_state'][t - 1] if t > 0 else [0.0] * h_size

        # 合并梯度
        if t == T - 1:
            dh = [delta_y * w_hy + dh_n for w_hy, dh_n in zip(W_hy, dh_next)]
        else:
            dh = dh_next.copy()
        dc = [dc_n + dh_i * o_t_i * tanh_derivative(c_t_i)
              for dc_n, dh_i, o_t_i, c_t_i in zip(dc_next, dh, o_t, c_t)]

        # 候选值梯度
        dc_hat = [dc_i * i_t_i * (1 - c_hat_t_i ** 2)
                 for dc_i, i_t_i, c_hat_t_i in zip(dc, i_t, c_hat_t)]

        # 输入门梯度
        di = [dc_i * c_hat_t_i * i_t_i * (1 - i_t_i)
             for dc_i, c_hat_t_i, i_t_i in zip(dc, c_hat_t, i_t)]

        # 遗忘门梯度
        df = [dc_i * c_prev_i * f_t_i * (1 - f_t_i)
             for dc_i, c_prev_i, f_t_i in zip(dc, c_prev, f_t)]

        # 输出门梯度
        do = [dh_i * math.tanh(c_t_i) * o_t_i * (1 - o_t_i)
             for dh_i, c_t_i, o_t_i in zip(dh, c_t, o_t)]

        # 计算各参数梯度
        x_t = x_sequence[t]
        for i in range(h_size):
            # 输入门参数
            grads['W_xi'][i] += di[i] * x_t
            grads['b_i'][i] += di[i]
            for j in range(h_size):
                grads['W_hi'][i][j] += di[i] * h_prev[j]

            # 遗忘门参数
            grads['W_xf'][i] += df[i] * x_t
            grads['b_f'][i] += df[i]
            for j in range(h_size):
                grads['W_hf'][i][j] += df[i] * h_prev[j]

            # 候选值参数
            grads['W_xc'][i] += dc_hat[i] * x_t
            grads['b_c'][i] += dc_hat[i]
            for j in range(h_size):
                grads['W_hc'][i][j] += dc_hat[i] * h_prev[j]

            # 输出门参数
            grads['W_xo'][i] += do[i] * x_t
            grads['b_o'][i] += do[i]
            for j in range(h_size):
                grads['W_ho'][i][j] += do[i] * h_prev[j]

        # 计算传递给前一步的梯度
        dh_prev = [0.0] * h_size
        for j in range(h_size):
            dh_prev[j] = sum(
                di[i] * W_hi[i][j] +
                df[i] * W_hf[i][j] +
                dc_hat[i] * W_hc[i][j] +
                do[i] * W_ho[i][j]
                for i in range(h_size)
            )

        dc_prev = [dc[i] * f_t[i] for i in range(h_size)]

        dh_next = dh_prev
        dc_next = dc_prev

        # 打印细节
        print(f"\n时间步 {t + 1} 梯度:")
        print(f"候选值梯度: {[round(x, 4) for x in dc_hat]}")
        print(f"输入门梯度: {[round(x, 4) for x in di]}")
        print(f"遗忘门梯度: {[round(x, 4) for x in df]}")
        print(f"输出门梯度: {[round(x, 4) for x in do]}")

    return grads


# 参数更新函数
def update_parameters(params, grads, lr):
    (W_xi, W_hi, b_i,
     W_xf, W_hf, b_f,
     W_xc, W_hc, b_c,
     W_xo, W_ho, b_o,
     W_hy, b_y) = params

    new_params = (
        [w - lr * dw for w, dw in zip(W_xi, grads['W_xi'])],
        [[w - lr * dw for w, dw in zip(row, grad_row)]
         for row, grad_row in zip(W_hi, grads['W_hi'])],
        [b - lr * db for b, db in zip(b_i, grads['b_i'])],
        [w - lr * dw for w, dw in zip(W_xf, grads['W_xf'])],
        [[w - lr * dw for w, dw in zip(row, grad_row)]
         for row, grad_row in zip(W_hf, grads['W_hf'])],
        [b - lr * db for b, db in zip(b_f, grads['b_f'])],
        [w - lr * dw for w, dw in zip(W_xc, grads['W_xc'])],
        [[w - lr * dw for w, dw in zip(row, grad_row)]
         for row, grad_row in zip(W_hc, grads['W_hc'])],
        [b - lr * db for b, db in zip(b_c, grads['b_c'])],
        [w - lr * dw for w, dw in zip(W_xo, grads['W_xo'])],
        [[w - lr * dw for w, dw in zip(row, grad_row)]
         for row, grad_row in zip(W_ho, grads['W_ho'])],
        [b - lr * db for b, db in zip(b_o, grads['b_o'])],
        [w - lr * dw for w, dw in zip(W_hy, grads['W_hy'])],
        b_y - lr * grads['b_y']
    )
    return new_params


# 训练配置
hidden_size = 2  # 为了简化演示使用较小的隐藏层
sequence_length = 3
learning_rate = 0.1
epochs = 50

# 初始化
data = generate_data(sequence_length)
params = initialize_parameters(hidden_size)

# 训练循环
for epoch in range(epochs):
    total_loss = 0
    for sequence, label in data:
        # 前向传播
        y_pred, gates = forward_pass(sequence, params)

        # 计算损失
        loss = - (label * math.log(y_pred + 1e-8) + (1 - label) * math.log(1 - y_pred + 1e-8))
        total_loss += loss

        # 反向传播
        grads = backward_pass(sequence, y_pred, label, gates, params)

        # 参数更新
        params = update_parameters(params, grads, learning_rate)

    # 打印训练进度
    if (epoch + 1) % 10 == 0:
        print(f"\nEpoch {epoch + 1}/{epochs}, 平均损失: {total_loss / len(data):.4f}")

# 测试示例
test_sequence = [1, 0, 1]
label = sum(test_sequence) % 2
y_pred, _ = forward_pass(test_sequence, params)
print(f"\n测试序列 {test_sequence} 的预测值: {y_pred:.4f}" )