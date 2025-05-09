import subprocess
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import zscore


class LidarVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.scatter = None
        self.current_elev = 45  # 初始仰角
        self.current_azim = 45  # 初始方位角
        self.zoom_factor = 1.0  # 缩放因子
        self.base_max_range = 5.0  # 基础坐标范围

        # 初始化3D坐标设置
        self.ax.set_xlabel('X Axis', fontsize=10)
        self.ax.set_ylabel('Y Axis', fontsize=10)
        self.ax.set_zlabel('Z Axis', fontsize=10)
        self.ax.view_init(elev=self.current_elev, azim=self.current_azim)
        self.ax.set_box_aspect([1, 1, 1])  # 等比例坐标轴
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.xaxis.pane.fill = False  # 透明背景
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        # 设置交互功能
        plt.ion()
        plt.show(block=False)

        # 绑定事件处理
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.is_dragging = False

    def on_mouse_move(self, event):
        """处理鼠标拖拽旋转"""
        if event.inaxes == self.ax and self.is_dragging:
            self.current_elev = self.ax.elev
            self.current_azim = self.ax.azim
            self.fig.canvas.draw_idle()

    def on_mouse_release(self, event):
        """鼠标释放事件"""
        self.is_dragging = False

    def on_scroll(self, event):
        """处理滚轮缩放事件"""
        if event.inaxes == self.ax:
            # 计算缩放因子（限制在0.5-3倍之间）
            scale_factor = 1.1 if event.button == 'up' else 0.9
            self.zoom_factor = np.clip(self.zoom_factor * scale_factor, 0.5, 3.0)
            self.fig.canvas.draw_idle()

    def _calculate_dynamic_range(self, points):
        """计算动态坐标范围"""
        max_range = max(points[:, 0].ptp(),  # X轴范围
                        points[:, 1].ptp(),  # Y轴范围
                        points[:, 2].ptp())  # Z轴范围
        max_range = max(max_range * 0.6 * self.zoom_factor, 0.1)  # 防止过小
        mid_x = np.median(points[:, 0])
        mid_y = np.median(points[:, 1])
        mid_z = np.median(points[:, 2])
        return mid_x, mid_y, mid_z, max_range

    def update_plot(self, points):
        """更新点云可视化"""
        if len(points) < 3:
            return

        points = np.array(points)
        current_elev = self.ax.elev
        current_azim = self.ax.azim

        # 计算动态坐标范围
        mid_x, mid_y, mid_z, max_range = self._calculate_dynamic_range(points)

        # 设置坐标范围
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # 更新或创建散点图
        if self.scatter is None:
            self.scatter = self.ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=points[:, 2],  # Z值作为颜色
                cmap='plasma',  # 色谱选择
                s=8,  # 点大小
                alpha=0.7,  # 透明度
                depthshade=True,  # 深度阴影
                picker=True  # 启用点选择
            )
        else:
            # 高效更新数据
            self.scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
            self.scatter.set_array(points[:, 2])  # 更新颜色数组

        # 保持视角不变
        self.ax.view_init(elev=current_elev, azim=current_azim)

        try:
            # 增量渲染更新
            self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(0.001)
        except Exception as e:
            print(f"渲染更新异常: {str(e)}")





def denoise_point_cloud(points, k=10, z_threshold=2.0):
    """
    基于统计的离群点去除
    :param points: 输入点云(N,3)
    :param k: 最近邻数量
    :param z_threshold: Z-score阈值
    :return: 过滤后的点云
    """
    if len(points) < k:
        return points

    tree = KDTree(points)
    distances = []
    for point in points:
        dists = tree.query(point, k=k + 1)[0]  # 包含自身
        avg_dist = np.mean(dists[1:])  # 排除自身
        distances.append(avg_dist)

    z_scores = np.abs(zscore(distances))
    mask = z_scores < z_threshold
    return points[mask]


def smooth_point_cloud(points, k=8, weight_factor=0.5):
    """
    基于邻域平均的平滑处理
    :param points: 输入点云(N,3)
    :param k: 最近邻数量
    :param weight_factor: 平滑权重(0-1)
    :return: 平滑后的点云
    """
    if len(points) < k:
        return points

    tree = KDTree(points)
    smoothed = []
    for point in points:
        indices = tree.query(point, k=k)[1]
        neighbors = points[indices]
        centroid = np.mean(neighbors, axis=0)
        # 加权平均
        smoothed_point = point * (1 - weight_factor) + centroid * weight_factor
        smoothed.append(smoothed_point)

    return np.array(smoothed)


if __name__ == '__main__':
    password = "future"

    # 初始化可视化器
    vis = LidarVisualizer()

    try:
        # 执行C的可执行文件
        process = subprocess.Popen(['sudo', '-S', './YT_lidar_V1.0'],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   bufsize=1)

        process.stdin.write(password + '\n')
        process.stdin.flush()

        now_id = None
        frameCount = 0
        allNumbers = []
        formerINFO = []
        bankINFO = []

        # 获取终端输出的数据并进行一系列的数据处理操作
        for line in process.stdout:
            numbers = []
            parts = line.split()
            # 转换为可以进行处理的数据
            for part in parts:
                try:
                    num = float(part)
                    numbers.append(num)
                except:
                    continue


            # 定期清理数据
            if len(allNumbers) >= 10000:
                allNumbers = allNumbers[-10000:]
            if len(formerINFO) >= 5000:
                formerINFO = formerINFO[-5000:]
            if len(bankINFO) >= 5000:
                bankINFO = bankINFO[-5000:]


            # 对数据进行处理
            if numbers:
                if now_id is None:
                    now_id = numbers[1]
                elif numbers[1] != now_id:

                    if frameCount % 3 == 0:     # 刷新率
                        # 转换为numpy数组
                        if len(allNumbers) > 0:
                            points = np.array(allNumbers)
                            formerPoints = np.array(formerINFO)
                            bankPoints = np.array(bankINFO)

                            if frameCount % 10 == 0:
                                # 降噪处理
                                if len(points) > 50:  # 当点数足够时处理
                                    points = denoise_point_cloud(points)
                                    formerPoints = denoise_point_cloud(formerPoints)
                                    bankPoints = denoise_point_cloud(bankPoints)

                                # 平滑处理
                                points = smooth_point_cloud(points)
                                formerPoints = smooth_point_cloud(formerPoints)
                                bankPoints = smooth_point_cloud(bankPoints)

                            # 数据裁剪
                            if len(points) > 5000:
                                points = points[-5000:]
                            if len(formerPoints) > 2500:
                                formerPoints = formerPoints[-2500:]
                            if len(bankPoints) > 2500:
                                bankPoints = bankPoints[-2500:]

                            # 更新可视化
                            # vis.update_plot(points)
                            vis.update_plot(formerPoints)
                            frameCount = 0


                    frameCount += 1
                    now_id = numbers[1]

                if len(numbers) >= 7:
                    coord = numbers[4:7]
                    if coord[1] >= 0:
                        formerINFO.append(coord)
                    elif coord[1] < 0:
                        bankINFO.append(coord)
                    if len(coord) == 3:
                        allNumbers.append(coord)


        for line in process.stderr:
            print("STDERR:", line.strip())

        process.wait()
        print(f"Process exited with code {process.returncode}")


    except Exception as e:
        print(f"Error: {e}")
    finally:
        if process:
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()
        plt.close('all')










