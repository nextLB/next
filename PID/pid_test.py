import dearpygui.dearpygui as dpg
from PID import PID  # 从自定义模块导入 PID 类

dpg.create_context()  # 创建 DearPyGui 运行上下文（必须的初始化操作）

# 创建测试数据的容器
datax_test = []  # X轴数据（时间/步长）
datay_test = []  # Y轴数据（PID输出值）


def update_data_test():
    global datax_test, datay_test
    # 从 GUI 控件获取参数值
    Sp = dpg.get_value("Sp")  # 获取设定值（整数滑动条）
    Kp = dpg.get_value("Kp")  # 获取比例系数（浮点滑动条）
    Ki = dpg.get_value("Ki")  # 获取积分系数（浮点滑动条）
    Kd = dpg.get_value("Kd")  # 获取微分系数（浮点滑动条）

    # 初始化数据（前5个点为0）
    datax_test = list(range(5))  # 创建 [0,1,2,3,4]
    datay_test = [0 for _ in range(5)]  # 创建 [0,0,0,0,0]

    # 创建 PID 实例（传入实时参数）
    pid = PID(Kp, Ki, Kd, Sp)

    # 模拟 PID 运行（生成95个数据点）
    for i in range(5, 100):
        datax_test.append(i)  # 添加时间步长（5-99）
        datay_test.append(pid.calculate())  # 计算并记录 PID 输出

    # 更新图表数据
    dpg.set_value("series_TestModel", [datax_test, datay_test])
    # 自动调整坐标轴范围（根据新数据自动缩放）
    dpg.fit_axis_data("x_axis_TestModel")  # X轴自适应
    dpg.fit_axis_data("y_axis_TestModel")  # Y轴自适应


# 创建主窗口（设置标签为"Primary Window"，高度填满父容器）
with dpg.window(tag="Primary Window", height=-1):
    # 创建图表主题（黄色线条+方形标记）
    with dpg.theme(tag="plot_theme_TestModel"):
        with dpg.theme_component(dpg.mvLineSeries):  # 针对折线图组件
            dpg.add_theme_color(
                dpg.mvPlotCol_Line, (255, 255, 50),  # 设置线条颜色为黄色
                category=dpg.mvThemeCat_Plots
            )
            dpg.add_theme_style(
                dpg.mvPlotStyleVar_Marker,
                dpg.mvPlotMarker_Square,  # 设置数据点为方形
                category=dpg.mvThemeCat_Plots
            )
            dpg.add_theme_style(
                dpg.mvPlotStyleVar_MarkerSize,
                3,  # 设置标记大小
                category=dpg.mvThemeCat_Plots
            )

    # 创建绘图区域（标签为"Simulate"，高度400，宽度自适应）
    with dpg.plot(label="Simulate", height=400, width=-1):
        dpg.add_plot_legend()  # 添加图例

        # 创建坐标轴（必须步骤）
        dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="x_axis_TestModel")  # X轴
        dpg.add_plot_axis(dpg.mvYAxis, label="y", tag="y_axis_TestModel")  # Y轴

        # 添加折线系列（绑定到Y轴，使用初始数据）
        dpg.add_line_series(
            datax_test, datay_test,
            parent="y_axis_TestModel",
            tag="series_TestModel"
        )

    # 将主题应用到折线系列
    dpg.bind_item_theme("series_TestModel", "plot_theme_TestModel")

    # 参数控制面板（使用水平布局组）
    with dpg.group(horizontal=True):
        dpg.add_text("Set Point")  # 标签文本
        dpg.add_slider_int(  # 整数滑动条（设定值）
            tag="Sp",
            default_value=350,  # 默认值
            max_value=500,
            width=-1,  # 宽度自适应
            callback=update_data_test  # 值改变时触发更新
        )

    # 其他参数同理（Kp/Ki/Kd）
    with dpg.group(horizontal=True):
        dpg.add_text("Kp")
        dpg.add_slider_float(
            tag="Kp",
            default_value=0.04,
            min_value=-1,  # 允许负值（逆向调节）
            max_value=1,
            width=-1,
            callback=update_data_test  # 关键回调函数
        )
    # Ki和Kd的滑动条结构类似（省略重复注释）
    with dpg.group(horizontal=True):
        dpg.add_text("Ki")
        dpg.add_slider_float(
            tag="Ki",
            default_value=0.7,
            min_value=-2,
            max_value=2,
            width=-1,
            callback=update_data_test,
        )

    with dpg.group(horizontal=True):
        dpg.add_text("Kd")
        dpg.add_slider_float(
            tag="Kd",
            default_value=0.015,
            min_value=-2,
            max_value=2,
            width=-1,
            callback=update_data_test,
        )


# 首次运行更新数据（生成初始曲线）
update_data_test()

# 创建视口（窗口容器）
dpg.create_viewport(
    title="PID Simulate",  # 窗口标题
    width=700,  # 初始宽度
    height=550  # 初始高度
)

# DearPyGui 初始化流程
dpg.setup_dearpygui()  # 初始化后端
dpg.show_viewport()  # 显示主视口
dpg.set_primary_window("Primary Window", True)  # 设置主窗口

# 进入主循环
dpg.start_dearpygui()

# 清理资源
dpg.destroy_context()