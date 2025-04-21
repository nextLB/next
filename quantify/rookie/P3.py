from pyecharts import options as opts
from pyecharts.charts import Kline
import akshare as ak
import pandas as pd

# 获取复权数据（后复权更准确）
stock_df = ak.stock_zh_a_hist(
    symbol="600519",  # 直接使用6位数字代码
    period="daily",
    start_date="20200101",
    end_date="20221231",
    adjust="hfq"  # 后复权
)

# 转换数据格式
kline_data = stock_df[['开盘', '收盘', '最低', '最高']].values.tolist()
date_list = pd.to_datetime(stock_df['日期']).dt.strftime('%Y-%m-%d').tolist()

# 创建K线图
kline = (
    Kline()
    .add_xaxis(date_list)
    .add_yaxis("贵州茅台", kline_data)
    .set_global_opts(
        title_opts=opts.TitleOpts(title="贵州茅台后复权K线图 (2020-2022)"),
        xaxis_opts=opts.AxisOpts(type_="category"),
        yaxis_opts=opts.AxisOpts(is_scale=True),
        datazoom_opts=[opts.DataZoomOpts(type_="inside")],
        toolbox_opts=opts.ToolboxOpts(
            feature={
                "dataZoom": {"yAxisIndex": "none"},
                "restore": {},
                "saveAsImage": {"pixel_ratio": 2},
            }
        )
    )
)

kline.render("maotai_kline_hfq.html")