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
def vec_add(v1, v2):
    return [x + y for x, y in zip(v1, v2)]


def scalar_vec_mult(s, v):
    return [s * x for x in v]


def mat_vec_mult(matrix, vector):
    return [sum(row[i] * vector[i] for i in range(len(vector))) for row in matrix]


def transpose(matrix):
    return list(map(list, zip(*matrix)))


def vec_tanh(v):
    return [math.tanh(x) for x in v]


def vec_tanh_derivative(v):
    return [1 - math.tanh(x) ** 2 for x in v]


# 参数初始化
def initialize_parameters(hidden_size):
    W_xh = [random.uniform(-0.01, 0.01) for _ in range(hidden_size)]
    W_hh = [[random.uniform(-0.01, 0.01) for _ in range(hidden_size)] for _ in range(hidden_size)]
    W_hy = [random.uniform(-0.01, 0.01) for _ in range(hidden_size)]
    bh = [random.uniform(-0.01, 0.01) for _ in range(hidden_size)]
    by = random.uniform(-0.01, 0.01)
    return W_xh, W_hh, W_hy, bh, by


# 前向传播（带详细输出）
def forward_pass(x_sequence, params):
    W_xh, W_hh, W_hy, bh, by = params
    h_states, a_states = [], []
    h_prev = [0.0] * len(W_xh)

    print("\n=== 前向传播开始 ===")
    for t in range(len(x_sequence)):
        x_t = x_sequence[t]
        # 计算加权和
        weighted_input = scalar_vec_mult(x_t, W_xh)
        weighted_hidden = mat_vec_mult(W_hh, h_prev)
        a_t = vec_add(vec_add(weighted_input, weighted_hidden), bh)
        # 应用tanh激活
        h_t = vec_tanh(a_t)

        # 保存中间状态
        h_states.append(h_t)
        a_states.append(a_t)

        # 打印细节
        print(f"\n时间步 {t + 1}/{len(x_sequence)}")
        print(f"输入值: {x_t}")
        print(f"前一个隐藏状态: {[round(x, 4) for x in h_prev]}")
        print(f"输入权重计算: {[round(x, 4) for x in weighted_input]}")
        print(f"隐藏权重计算: {[round(x, 4) for x in weighted_hidden]}")
        print(f"偏置相加后: {[round(x, 4) for x in a_t]}")
        print(f"新隐藏状态: {[round(x, 4) for x in h_t]}")

        h_prev = h_t

    # 最终输出计算
    z = sum(w * h for w, h in zip(W_hy, h_states[-1])) + by
    y_pred = 1 / (1 + math.exp(-z))
    print("\n最终输出计算:")
    print(f"隐藏到输出权重: {[round(x, 4) for x in W_hy]}")
    print(f"最终隐藏状态: {[round(x, 4) for x in h_states[-1]]}")
    print(f"输出偏置: {round(by, 4)}")
    print(f"预测值: {round(y_pred, 4)} (实际标签: {label})")

    return y_pred, h_states, a_states


# 反向传播（带详细输出）
def backward_pass(x_sequence, y_pred, label, h_states, a_states, params):
    W_xh, W_hh, W_hy, bh, by = params
    hidden_size = len(W_xh)

    print("\n=== 反向传播开始 ===")

    # 初始化梯度
    dW_xh = [0.0] * hidden_size
    dW_hh = [[0.0] * hidden_size for _ in range(hidden_size)]
    dW_hy = [0.0] * hidden_size
    dbh = [0.0] * hidden_size
    dby = 0.0

    # 输出层梯度
    delta_output = y_pred - label
    print(f"\n输出层梯度: {round(delta_output, 4)}")

    # 隐藏到输出权重的梯度
    dW_hy = scalar_vec_mult(delta_output, h_states[-1])
    dby = delta_output

    # 初始化反向传播的隐藏状态梯度
    dh_next = [0.0] * hidden_size

    for t in reversed(range(len(x_sequence))):
        a_t = a_states[t]
        h_t = h_states[t]

        # 计算当前梯度
        if t == len(x_sequence) - 1:
            # 最后一个时间步需要加上输出梯度
            dh = scalar_vec_mult(delta_output, W_hy)
            dh = vec_add(dh, dh_next)
        else:
            dh = dh_next

        # 应用tanh导数
        tanh_deriv = vec_tanh_derivative(a_t)
        dh = [dh[i] * tanh_deriv[i] for i in range(hidden_size)]

        # 获取前一个隐藏状态
        h_prev = h_states[t - 1] if t > 0 else [0.0] * hidden_size

        # 计算各参数梯度
        x_t = x_sequence[t]
        for i in range(hidden_size):
            # 输入权重梯度
            dW_xh[i] += dh[i] * x_t
            # 隐藏偏置梯度
            dbh[i] += dh[i]
            # 隐藏到隐藏权重梯度
            for j in range(hidden_size):
                dW_hh[i][j] += dh[i] * h_prev[j]

        # 计算传递给前一个时间步的梯度
        dh_next = mat_vec_mult(transpose(W_hh), dh)

        # 打印当前时间步梯度
        print(f"\n时间步 {t + 1} 梯度:")
        print(f"当前隐藏梯度: {[round(x, 4) for x in dh]}")
        print(f"输入权重梯度更新: {[round(x, 4) for x in dW_xh]}")
        print(f"隐藏到隐藏梯度更新: {[[round(x, 4) for x in row] for row in dW_hh]}")

    return dW_xh, dW_hh, dW_hy, dbh, dby


# 参数更新函数
def update_parameters(params, grads, lr):
    W_xh, W_hh, W_hy, bh, by = params
    dW_xh, dW_hh, dW_hy, dbh, dby = grads

    new_W_xh = [w - lr * dw for w, dw in zip(W_xh, dW_xh)]
    new_W_hh = [[w - lr * dw for w, dw in zip(row, d_row)]
                for row, d_row in zip(W_hh, dW_hh)]
    new_W_hy = [w - lr * dw for w, dw in zip(W_hy, dW_hy)]
    new_bh = [b - lr * db for b, db in zip(bh, dbh)]
    new_by = by - lr * dby

    return (new_W_xh, new_W_hh, new_W_hy, new_bh, new_by)


# 训练配置
hidden_size = 3
sequence_length = 3
learning_rate = 0.1
epochs = 100

# 初始化
data = generate_data(sequence_length)
params = initialize_parameters(hidden_size)

# 训练循环
for epoch in range(epochs):
    total_loss = 0
    for sequence, label in data:
        # 前向传播
        y_pred, h_states, a_states = forward_pass(sequence, params)

        # 计算损失
        loss = - (label * math.log(y_pred + 1e-8) + (1 - label) * math.log(1 - y_pred + 1e-8))
        total_loss += loss

        # 反向传播
        grads = backward_pass(sequence, y_pred, label, h_states, a_states, params)

        # 参数更新
        params = update_parameters(params, grads, learning_rate)

    # 打印训练进度
    if (epoch + 1) % 10 == 0:
        print(f"\nEpoch {epoch + 1}/{epochs}, 平均损失: {total_loss / len(data):.4f}")

# 测试示例
test_sequence = [1, 0, 1]
label = sum(test_sequence) % 2
y_pred, _, _ = forward_pass(test_sequence, params)
print(f"\n测试序列 {test_sequence} 的预测值: {y_pred:.4f} (实际标签: {label})")