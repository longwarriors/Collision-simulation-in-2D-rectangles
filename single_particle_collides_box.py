# -*- coding: utf-8 -*-
# @Author: Zhangxiaoxu
# @Date:   2023-03-15 15:27:12
# @Last Modified by:   longwarriors
# @Last Modified time: 2023-03-16 17:35:54
import numpy as np
import torch
from torch import tensor
from matplotlib import animation
import matplotlib.pyplot as plt
plt.style.use(["science", "notebook", "grid"])


"""参数设置"""
radius = 0.02  # 圆盘半径
bound_l, bound_r = 0.0, 1.0  # 包围盒左右边界
bound_d, bound_u = 0.0, 1.0  # 包围盒上下边界
n_incres = 80000  # dicrete time increments
t_end = 5  # 秒
dt = t_end / n_incres
t_nodes = torch.linspace(
    start=0,
    end=t_end,
    steps=n_incres + 1)  # my convention
speed = 28
coords = torch.empty((n_incres + 1, 2))
velocs = torch.empty((n_incres + 1, 2))
coords[0] = torch.rand(2)  # 随机给一个初始位置
direction = 1 - 2 * torch.rand(2)
direction = direction / direction.norm()
velocs[0] = speed * direction  # 随机给一个初始速度矢量

for n in range(n_incres):
    """碰撞检测
    竖边碰撞条件 (x < left + r & vx < 0) | (x > right - r & vx > 0)
    横边碰撞条件 (y < down + r & vy < 0) | (y > up -r & vy > 0)
    """
    collisions_vertical = ((coords[n, 0] < bound_l + radius) & (velocs[n, 0] < 0)) | (
        (coords[n, 0] > bound_r - radius) & (velocs[n, 0] > 0))
    collisions_horizon = ((coords[n, 1] < bound_d + radius) & (velocs[n, 1] < 0)) | (
        (coords[n, 1] > bound_u - radius) & (velocs[n, 1] > 0))

    # 撞竖直围栏则vx反号，撞水平围栏则vy反号
    if collisions_vertical:
        velocs[n, 0] = velocs[n, 0] * -1
        velocs[n + 1] = velocs[n]
    elif collisions_horizon:
        velocs[n, 1] = velocs[n, 1] * -1
        velocs[n + 1] = velocs[n]
    else:
        velocs[n + 1] = velocs[n]

    """按新速度更新粒子位置"""
    coords[n + 1] = coords[n] + velocs[n] * dt


"""每一帧新增一个点"""
# def create_animation():
# 创建画板
fig = plt.figure(figsize=(8, 6))
fig.set_tight_layout(True)  # 紧凑组排
fig.text(x=0.8, y=0.04, s="Designed by longwarriors",
         style="italic", fontsize=8, color="red")


# 创建画布
ax1 = plt.subplot(1, 1, 1)
ax1.set_xlim(0.0, 1.0)
ax1.set_ylim(0.0, 1.0)
ax1.set_title("Particle dynamic")
ax1.set_xlabel("Coordinate, $x$")
ax1.set_ylabel("Coordinate, $y$")


# 创建画布的计时器
timer = ax1.text(0.8, 0.5, "", fontsize=15,
                 transform=ax1.transAxes,
                 bbox=dict(facecolor="white", edgecolor="black")
                 )  # text实例


# 创建画布的图线
# creating an empty plot element
(line1,) = ax1.plot([], [], linewidth=3, color="cornflowerblue")
(dot1,) = ax1.plot([], [], 'yo', markersize=10, markeredgecolor='r')

# 清空当前帧 initialize


def init():
    # creating an empty plot/frame
    timer.set_text("")
    line1.set_data([], [])
    dot1.set_data([], [])
    return timer, line1, dot1


# 更新新一帧的数据 refresh
def update(n):  # time[n]
    # set/update the axes data
    timer.set_text("time = {:.3f}".format(n * dt))
    line1.set_data(coords[:n, 0], coords[:n, 1])
    dot1.set_data(coords[n, 0], coords[n, 1])
    return timer, line1, dot1


# perform animation
ani = animation.FuncAnimation(
    fig, update,
    frames=np.arange(0, n_incres + 1, 20, dtype=int),
    interval=0,  # 帧之间的延迟（毫秒）默认为200
    blit=True,  # 是否执行blitting优化
    init_func=init,
)


# show animation
plt.show()
