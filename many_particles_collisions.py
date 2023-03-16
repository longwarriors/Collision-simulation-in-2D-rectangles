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
n_particles = 600  # 粒子数量
n_incres = 80000  # dicrete time increments
t_end = 5  # 秒
dt = t_end / n_incres
t_nodes = torch.linspace(
    start=0,
    end=t_end,
    steps=n_incres + 1)  # my convention
speed = 28


coords = torch.empty((n_incres + 1, n_particles, 2))
velocs = torch.empty((n_incres + 1, n_particles, 2))
