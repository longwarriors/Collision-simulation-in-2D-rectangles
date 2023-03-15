# -*- coding: utf-8 -*-
# @Author: Zhangxiaoxu
# @Date:   2023-03-15 15:27:12
# @Last Modified by:   longwarriors
# @Last Modified time: 2023-03-15 19:03:18
import numpy as np
import torch
from torch import tensor
from matplotlib import animation
import matplotlib.pyplot as plt
plt.style.use(["science", "notebook", "grid"])


"""每一帧新增一个点"""
# def create_animation():
# 创建画板
fig = plt.figure(figsize=(8, 6))
fig.set_tight_layout(True)  # 紧凑组排
fig.text(x=0.8, y=0.04, s="Designed by longwarriors", style="italic", fontsize=8, color="red")


# 创建画布
ax1 = plt.subplot(1, 1, 1)
ax1.set_xlim(0.0, 1.0)
ax1.set_ylim(0.0, 1.0)
ax1.set_title("Particle dynamic")
ax1.set_xlabel("Coordinate, $x$")
ax1.set_ylabel("Coordinate, $y$")


# 创建画布的计时器
timer = ax1.text(1.0, 0.5, "", fontsize=15,
    transform=ax1.transAxes,
    bbox=dict(facecolor="white", edgecolor="black")
    )  # text实例


# 创建画布的图线
(line1,) = ax1.plot([], [], linewidth=3, color="cornflowerblue")  # creating an empty plot element


# 清空当前帧 initialize
def init():
    # creating an empty plot/frame
    timer.set_text("")
    line1.set_data([], [])
    return line1


# 更新新一帧的数据 refresh
def update(n):  # time[n]
    # set/update the axes data
    timer.set_text("time = {:.2f}".format(n*tStep))
    xs.append()
    ys.append()
    line1.set_data(xs, ys)
    return line1


# perform animation
ani = animation.FuncAnimation(
    fig, update,
    frames=np.arange(0, len(ts), 1, dtype=int),
    interval=1,  # 帧之间的延迟（毫秒）默认为200
    blit=False,  # 是否执行blitting优化
    init_func=init,
)



# show animation
plt.show()
