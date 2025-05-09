
import subprocess
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

password = "future"


class LidarVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.scatter = None
        self.current_elev = 45
        self.current_azim = 45

        # 初始化3D设置
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.set_zlabel('Z Axis')
        self.ax.view_init(elev=self.current_elev, azim=self.current_azim)
        self.ax.set_box_aspect([1, 1, 1])  # 等比例坐标
        self.ax.grid(True, linestyle='--', alpha=0.5)
        plt.ion()
        plt.show(block=False)

        # 绑定鼠标事件
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.is_dragging = False

    def on_mouse_move(self, event):
        if event.inaxes == self.ax and self.is_dragging:
            self.current_elev = self.ax.elev
            self.current_azim = self.ax.azim
            self.fig.canvas.draw_idle()

    def on_mouse_release(self, event):
        self.is_dragging = False

    def update_plot(self, points):
        if len(points) < 3:
            return

        points = np.array(points)
        current_elev = self.ax.elev
        current_azim = self.ax.azim

        # 动态调整坐标范围（等比例）
        max_range = max(points[:, 0].ptp(),
                        points[:, 1].ptp(),
                        points[:, 2].ptp()) * 0.6
        mid_x = np.median(points[:, 0])
        mid_y = np.median(points[:, 1])
        mid_z = np.median(points[:, 2])

        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # 更新散点数据
        if self.scatter is None:
            self.scatter = self.ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=points[:, 2], cmap='plasma', s=8,
                alpha=0.8, depthshade=True
            )
        else:
            self.scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
            self.scatter.set_array(points[:, 2])

        # 恢复视角
        self.ax.view_init(elev=current_elev, azim=current_azim)

        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(0.001)
        except:
            pass


# 初始化可视化器
vis = LidarVisualizer()

try:
    process = subprocess.Popen(['sudo', '-S', './YT_lidar_V1.0'],
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True,
                               bufsize=1)

    process.stdin.write(password + '\n')
    process.stdin.flush()

    now_id = None
    frame_count = 0
    all_numbers = []

    for line in process.stdout:
        numbers = []
        parts = line.split()
        for part in parts:
            try:
                num = float(part)
                numbers.append(num)
            except:
                continue

        if numbers:
            if now_id is None:
                now_id = numbers[1]
            elif numbers[1] != now_id:
                if frame_count % 3 == 0:  # 提高刷新率
                    vis.update_plot(all_numbers)
                    frame_count = 0
                frame_count += 1
                if len(all_numbers) > 5000:  # 控制数据量
                    all_numbers = all_numbers[-2000:]
                now_id = numbers[1]
            if len(numbers) >= 7:
                coord = numbers[4:7]
                if len(coord) == 3:
                    all_numbers.append(coord)

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



