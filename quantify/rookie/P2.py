import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt

# 获取数据 (茅台A股)
symbol = "600519"
start_date = "20190101"  # akshare日期格式为YYYYMMDD
end_date = "20210101"

# 获取历史数据
data = ak.stock_zh_a_hist(
    symbol=symbol,
    period="daily",
    start_date=start_date,
    end_date=end_date,
    adjust="hfq"  # 后复权
).rename(columns={
    '日期': 'Date',
    '开盘': 'Open',
    '收盘': 'Close',
    '最高': 'High',
    '最低': 'Low',
    '成交量': 'Volume'
}).set_index('Date')
data.index = pd.to_datetime(data.index)

# 计算移动平均
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# 生成信号
data['Signal'] = 0
data.loc[data['SMA_50'] > data['SMA_200'], 'Signal'] = 1
data.loc[data['SMA_50'] < data['SMA_200'], 'Signal'] = -1

# 计算收益
data['Daily_Return'] = data['Close'].pct_change()
data['Strategy_Return'] = data['Signal'].shift(1) * data['Daily_Return']
data['Cumulative_Return'] = (1 + data['Strategy_Return']).cumprod()

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(data['Cumulative_Return'], label='策略收益', color='b')
plt.plot(data['Close'] / data['Close'].iloc[0], label='股价收益', color='g', alpha=0.5)
plt.title("双均线策略收益对比 (贵州茅台)")
plt.xlabel("日期")
plt.ylabel("累计收益率")
plt.legend()
plt.show()