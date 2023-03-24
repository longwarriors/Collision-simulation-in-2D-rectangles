# -*- coding: utf-8 -*-
# @Author: Zhangxiaoxu
# @Date:   2023-03-24 10:18:59
# @Last Modified by:   longwarriors
# @Last Modified time: 2023-03-24 10:19:57

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation 

plt.style.use(["science", "notebook", "grid"])

ts = 200
data = np.random.randn(ts, 20000)
n_segment = 88 # n_interval 统计的区间数
bin_edges = np.linspace(-4, 4, n_segment + 1) # edges or vertices


# 创建画板
fig = plt.figure(figsize=(8, 4))
fig.set_tight_layout(True)  # 紧凑组排
fig.text(
    x=0.8, y=0.04, s="Designed by longwarriors", style="italic", fontsize=8, color="red"
)


# 创建画布
ax1 = plt.subplot(1, 1, 1)
ax1.set_ylim(top=1.0)
_, _, bar_container = ax1.hist(data[0], bin_edges, lw=1, ec="yellow", fc="green", alpha=0.5)
timer = ax1.text(
    0.8,
    0.5,
    "",
    fontsize=15,
    transform=ax1.transAxes,
    bbox=dict(facecolor="white", edgecolor="black"),
)  # text实例

def update(frame_number):
    hist, _ = np.histogram(data[frame_number], bin_edges, density=True) # _ = bin_edges
    for count, rect in zip(hist, bar_container.patches):
        rect.set_height(count)
    timer.set_text("time = {:.3f}".format(frame_number))
    return bar_container.patches, timer

ani = animation.FuncAnimation(fig, update, 50, repeat=False, blit=True)
plt.show()