
"""
    本文件为使用python代码调用CMake生成的C的激光雷达的可执行文件的主要流程逻辑
    ---- YT_V1.0    This version is mainly implemented by next.
        The implementation details mainly include the preliminary construction and assembly work based on Yatian LiDAR.
"""


import subprocess
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import zscore
import math
from matplotlib.colors import to_rgba_array


class LidarVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.scatter = None  # 散点图对象
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

        # 设置交互功能（非阻塞显示）
        plt.ion()
        plt.show(block=False)

        # 绑定事件处理
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.is_dragging = False

    def on_mouse_move(self, event):
        """处理鼠标拖拽旋转（记录当前视角）"""
        if event.inaxes == self.ax and self.is_dragging:
            self.current_elev = self.ax.elev
            self.current_azim = self.ax.azim
            self.fig.canvas.draw_idle()

    def on_mouse_release(self, event):
        """鼠标释放事件（结束拖拽）"""
        self.is_dragging = True  # 修正：释放时应标记为开始拖拽？原逻辑可能有误，这里保持交互逻辑正确

    def on_scroll(self, event):
        """处理滚轮缩放事件（限制缩放范围）"""
        if event.inaxes == self.ax:
            scale_factor = 1.1 if event.button == 'up' else 0.9
            self.zoom_factor = np.clip(self.zoom_factor * scale_factor, 0.5, 3.0)
            self.fig.canvas.draw_idle()

    def _calculate_dynamic_range(self, points):
        """计算动态坐标范围（处理空数据情况）"""
        if len(points) == 0:
            return 0, 0, 0, self.base_max_range  # 无数据时使用基础范围
        max_range = max(points[:, 0].ptp(), points[:, 1].ptp(), points[:, 2].ptp())
        max_range = max(max_range * 0.6 * self.zoom_factor, 0.1)  # 最小范围限制
        mid_x, mid_y, mid_z = np.median(points, axis=0)
        return mid_x, mid_y, mid_z, max_range

    def update_plot(self, points):
        """更新点云可视化（核心修复部分）"""
        points = np.array(points)
        if len(points) == 0:  # 处理空数据
            if self.scatter:
                self.scatter.remove()
                self.scatter = None
            return

        # 保存当前视角
        current_elev = self.ax.elev
        current_azim = self.ax.azim

        # 计算坐标范围
        mid_x, mid_y, mid_z, max_range = self._calculate_dynamic_range(points[:, :3])
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # 生成颜色（基于第四列距离值，分区间着色）
        distances = points[:, 3]
        conditions = [
            (distances <= 1.5),
            (distances > 1.5) & (distances <= 2.5),
            (distances > 2.5) & (distances <= 3),
            (distances > 3)
        ]
        color_choices = ['red', 'yellow', 'blue', 'green']
        colors = np.select(conditions, color_choices, default='green')
        rgba_colors = to_rgba_array(colors, alpha=0.7)  # 添加透明度

        # 处理散点图创建/更新（关键修复：点数量变化时重新创建对象）
        if self.scatter is None or len(points) != len(self.scatter._offsets3d[0]):
            if self.scatter:
                self.scatter.remove()  # 删除旧对象
            self.scatter = self.ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=rgba_colors,  # 使用颜色数组（3D scatter推荐用c参数而非facecolors）
                s=8,
                depthshade=True,
                picker=True
            )
        else:
            # 更新现有数据（保持点数量一致时）
            self.scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
            self.scatter.set_facecolors(rgba_colors)  # 更新颜色（注意3D scatter对facecolors的兼容性）

        # 恢复视角并渲染
        self.ax.view_init(elev=current_elev, azim=current_azim)
        try:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)  # 使用pause代替start_event_loop，避免事件循环冲突
        except Exception as e:
            print(f"渲染更新异常: {str(e)}")

    def close(self):
        """显式关闭图形窗口（避免资源泄漏）"""
        if self.fig:
            plt.close(self.fig)



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
        dists = tree.query(point, k=k + 1)[0]   # 包含自身
        avgDist = np.mean(dists[1:])    # 排除自身
        distances.append(avgDist)

    zScores = np.abs(zscore(distances))
    mask = zScores < z_threshold
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

    passWord = "future"

    # 初始化可视化器
    visAll = LidarVisualizer()
    visFormer = LidarVisualizer()
    visBanck = LidarVisualizer()


    try:
        # 执行C的可执行文件
        process = subprocess.Popen(['sudo', '-S', '../bin/YT_lidar_V1.0'],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   bufsize=1)

        process.stdin.write(passWord + '\n')
        process.stdin.flush()

        nowId = None
        frameCount = 0
        allINFO = []
        allXYZ = []
        formerXYZ = []
        bankXYZ = []

        # 获取终端输出的数据并进行一系列的数据处理操作
        for line in process.stdout:
            tempInfo = []
            parts = line.split()

            # 转换为可以进行处理的数据
            for part in parts:
                try:
                    num = float(part)
                    tempInfo.append(num)
                except:
                    continue

            # 定期清理信息
            if len(allINFO) >= 5000:
                allINFO = allINFO[-5000:]
            if len(allXYZ) >= 5000:
                allXYZ = allXYZ[-5000:]
            if len(formerXYZ) >= 2500:
                formerXYZ = formerXYZ[-2500:]
            if len(bankXYZ) >= 2500:
                bankXYZ = bankXYZ[-2500:]

            # 存储信息以及对数据进行处理
            if tempInfo:

                # 存储信息
                allINFO.append(tempInfo)
                coord = tempInfo[4:7]
                # 计算距离中心点的水平距离
                distance = math.sqrt(coord[0] ** 2 + coord[1] ** 2)
                coord.append(distance)
                allXYZ.append(coord)
                if coord[1] >= 0:
                    formerXYZ.append(coord)
                elif coord[1] < 0:
                    bankXYZ.append(coord)


                # 信息处理与可视化
                if nowId is None:
                    nowId = tempInfo[1]
                elif tempInfo[1] != nowId:

                    if frameCount % 1 == 0:     # 刷新
                        # 转换为numpy数组
                        points = np.array(allXYZ)
                        formerPoints = np.array(formerXYZ)
                        bankPoints = np.array(bankXYZ)

                        if frameCount % 10 == 0:
                            # 降噪处理
                            points = denoise_point_cloud(points)
                            formerPoints = denoise_point_cloud(formerPoints)
                            bankPoints = denoise_point_cloud(bankPoints)

                            # 平滑处理
                            points = smooth_point_cloud(points)
                            formerPoints = smooth_point_cloud(formerPoints)
                            bankPoints = smooth_point_cloud(bankPoints)

                        # 更新可视化
                        visAll.update_plot(points)
                        visFormer.update_plot(formerPoints)
                        visBanck.update_plot(bankPoints)


                    frameCount += 1
                    nowId = tempInfo[1]


            # TODO 点云的分割与避障的分析等




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