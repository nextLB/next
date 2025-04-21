class PID:
    def __init__(self, kp, ki, kd, exp_val=0, dt=1):
        self.dt = dt              # 采样时间间隔（默认为1）
        self.KP = kp              # 比例系数
        self.KI = ki              # 积分系数
        self.KD = kd              # 微分系数
        self.exp_val = exp_val    # 目标值/设定值
        self.now_val = 0          # 当前实际值
        self.sum_err = 0          # 累计误差（用于积分项）
        self.now_err = 0          # 当前误差
        self.last_err = 0         # 上一次误差（用于微分项）

    def calculate(self):
        self.last_err = self.now_err  # 保存上一次误差
        self.now_err = self.exp_val - self.now_val  # 计算新误差
        self.sum_err += self.now_err  # 累计误差（积分项）

        # PID 计算公式
        self.now_val = (
            self.KP * self.now_err  # 比例项
            + self.KI * self.sum_err  # 积分项
            + self.KD * (self.now_err - self.last_err)  # 微分项
        )
        return self.now_val