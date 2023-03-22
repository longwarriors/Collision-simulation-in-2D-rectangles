import numpy as np
import torch
from torch import tensor
from matplotlib import animation
import matplotlib.pyplot as plt

plt.style.use(["science", "notebook", "grid"])

engine = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""参数设置"""
radius = 0.02  # 圆盘半径
speed = 17.5  # 粒子运动速率
bound_l, bound_r = -1.5, 1.5  # 包围盒左右边界
bound_d, bound_u = -1.5, 1.5  # 包围盒上下边界
n_particles = 150  # 粒子数量
n_increments = 20000  # dicrete time increments
t_end = 3  # 秒
dt = t_end / n_increments
t_nodes = torch.linspace(start=0, end=t_end, steps=n_increments + 1, device=engine)  # my convention


# 核心目标是求解坐标张量和速度张量
coords = torch.empty(size=(n_increments + 1, n_particles, 2), device=engine)
velocs = torch.empty(size=(n_increments + 1, n_particles, 2), device=engine)


def uniform_points(low_bounds: torch.tensor, high_bounds: torch.tensor, n_points):
    """在区域内产生均匀分布的点 [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2], ......]
    numpy.random.uniform 的每个维度的边界都一样不能改
    low_bounds  = [x_low,  y_low,  z_low]
    high_bounds = [x_high, y_high, z_high]
    randX = (x_high - x_low) * torch.rand() + x_low
    randY = (y_high - y_low) * torch.rand() + y_low
    randZ = (z_high - z_low) * torch.rand() + z_low
    """
    rand_mat = torch.rand(size=(n_points, len(low_bounds)))
    coef_mat = torch.diag(high_bounds - low_bounds)
    u_points = rand_mat @ coef_mat + low_bounds
    return u_points


"""初始化t=0时的坐标和速度"""
coords[0] = uniform_points(
    low_bounds=tensor([bound_l, bound_d]),
    high_bounds=tensor([bound_r, bound_u]),
    n_points=n_particles,
)  # 随机给一组初始坐标
directions = 1 - 2 * torch.rand(n_particles, 2)
directions = directions / directions.norm(p=2, dim=1).view(-1, 1)
# directions = torch.nn.functional.normalize(directions, p=2, dim=1)
velocs[0] = speed * directions  # 随机给一个初始速度矢量


"""相互碰撞检测"""
id_particles = torch.arange(n_particles, device=engine)  # 粒子编号构成的vector
id_pairs = torch.combinations(id_particles, 2)  # 粒子编号的组合构成的matrix
id_pairs_vec = id_pairs.flatten()  # 粒子编号的组合构成的vector
n_pairs = len(id_pairs)  # 需要考虑的相互作用数目


"""按步串行演化"""
for n in range(n_increments):

    """边界碰撞检测
    竖边碰撞条件 (x < left + r & vx < 0) | (x > right - r & vx > 0)
    横边碰撞条件 (y < down + r & vy < 0) | (y > up -r & vy > 0)
    """
    collisions_vertical = (
        (coords[n, :, 0] < bound_l + radius) & (velocs[n, :, 0] < 0)
    ) | ((coords[n, :, 0] > bound_r - radius) & (velocs[n, :, 0] > 0))
    collisions_horizon = (
        (coords[n, :, 1] < bound_d + radius) & (velocs[n, :, 1] < 0)
    ) | ((coords[n, :, 1] > bound_u - radius) & (velocs[n, :, 1] > 0))


    """把撞围栏的粒子速度提取出来"""
    velocs_reflect_vertical = velocs[n][collisions_vertical]
    velocs_reflect_horizon = velocs[n][collisions_horizon]


    """撞竖直围栏则vx反号，撞水平围栏则vy反号"""
    velocs_reflect_vertical[:, 0] = velocs_reflect_vertical[:, 0] * -1  # -vx
    velocs_reflect_horizon[:, 1] = velocs_reflect_horizon[:, 1] * -1  # -vy


    """把经过修改的速度刷新回去"""
    velocs[n][collisions_vertical] = velocs_reflect_vertical
    velocs[n][collisions_horizon] = velocs_reflect_horizon


    """相互碰撞检测"""
    coords_pairs = coords[n].index_select(0, id_pairs_vec)  # 按组合编号vector索引出来的粒子坐标构成的矩阵
    velocs_pairs = velocs[n].index_select(0, id_pairs_vec)  # 按组合编号vector索引出来的粒子速度构成的矩阵
    coords_pairs = coords_pairs.view(-1, 2, 2)  # 粒子的坐标组合构成的张量
    velocs_pairs = velocs_pairs.view(-1, 2, 2)  # 粒子的速度组合构成的张量

    dx_dy = coords_pairs.diff(dim=1)  # 坐标组合张量第二个维度做差得到 [x_i - x_j, y_i - y_j]
    dx_dy.squeeze_()  # 压缩所有值为1的维度成为matrix
    distance_pairs = dx_dy.norm(dim=-1)  # 粒子的坐标组合的距离vector
    collisions_other = distance_pairs < (2 * radius)  # 此刻相互距离小于2r的小球组合在下一刻被弹开
    if n > 0: # 相互碰撞条件 dr[n] < 2*r & dr[n] < dr[n-1]
        coords_pairs_before = coords[n-1].index_select(0, id_pairs_vec)
        coords_pairs_before = coords_pairs_before.view(-1, 2, 2)
        dx_dy_before = coords_pairs_before.diff(dim=1)
        dx_dy_before.squeeze_()
        distance_pairs_before = dx_dy_before.norm(dim=-1)
        collisions_other = collisions_other & (distance_pairs < distance_pairs_before)


    """把相互碰撞的粒子坐标和速度提取出来"""
    coords_collided_pairs = coords_pairs[collisions_other]  # 碰撞组合的坐标张量，如果没有碰撞则size=(0, 2, 2)
    velocs_collided_pairs = velocs_pairs[collisions_other]  # 碰撞组合的速度张量，如果没有碰撞则size=(0, 2, 2)
    dr_collided_pairs = coords_collided_pairs.diff(dim=1).squeeze()  # 碰撞组合的相对位置matrix，如果没有碰撞则size=(0, 2)
    dv_collided_pairs = velocs_collided_pairs.diff(dim=1).squeeze()  # 碰撞组合的相对速度matrix，如果没有碰撞则size=(0, 2)


    """相互碰撞后的粒子速度"""
    k_collided_pairs = (dr_collided_pairs * dv_collided_pairs).sum(-1) # 内积 torch.einsum("ij,ij->i", dr_collided_pairs, dv_collided_pairs)
    k_collided_pairs = k_collided_pairs / distance_pairs[collisions_other] ** 2
    k_collided_pairs = k_collided_pairs.unsqueeze(dim=1) * dr_collided_pairs
    # k_collided_pairs = torch.diag(k_collided_pairs) @ dr_collided_pairs
    # k_collided_pairs = torch.einsum("i,ij->ij", k_collided_pairs, dr_collided_pairs)

    velocs_collided_pairs[:, 0] = velocs_collided_pairs[:, 0] + k_collided_pairs
    velocs_collided_pairs[:, 1] = velocs_collided_pairs[:, 1] - k_collided_pairs
    id_collided = id_pairs[collisions_other].flatten()  # 碰撞粒子组合的编号
    velocs_new = velocs_collided_pairs.flatten(0, 1)  # list of vectors
    velocs[n].index_copy_(0, id_collided, velocs_new)  # 按碰撞的粒子编号替换新速度矢量


    """按新速度更新粒子位置"""
    coords[n + 1] = coords[n] + velocs[n] * dt
    velocs[n + 1] = velocs[n]


"""每一帧新增内容逐步显现"""
data_coords = coords.cpu().numpy()
data_velocs = velocs.norm(dim=2).cpu().numpy()
hist_bins = np.linspace(0, 1500, 10)

# 创建画板
fig = plt.figure(figsize=(8, 8))
fig.set_tight_layout(True)  # 紧凑组排
fig.text(
    x=0.8, y=0.04, s="Designed by longwarriors", style="italic", fontsize=8, color="red"
)


# 创建画布
ax1 = plt.subplot(1, 2, 1)
ax1.set_xlim(bound_l, bound_r)
ax1.set_ylim(bound_d, bound_u)
ax1.set_title("Particles collisions")
ax1.set_xlabel("Coordinate, $x$")
ax1.set_ylabel("Coordinate, $y$")

ax2 = plt.subplot(1, 2, 2)
ax2.set_title("Frequency statistics")
ax2.set_xlabel("Velocity, $[m/s]$")
ax2.set_ylabel("Proportion, $%$")


# 创建画布的计时器
timer = ax1.text(
    0.8,
    0.5,
    "",
    fontsize=15,
    transform=ax1.transAxes,
    bbox=dict(facecolor="white", edgecolor="black"),
)  # text实例


# 创建画布的图线 creating empty plots
# lines = []
dots = []
(_, _, hist_obj) = ax2.hist([], bins=int(n_particles/5), density=True)
for _ in range(n_particles):
    # line, = ax1.plot([], [], linewidth=3, color="cornflowerblue")    
    # lines.append(line)
    (dot,) = ax1.plot([], [], "yo", markersize=10, markeredgecolor="r")
    dots.append(dot)


# 清空当前帧 initialize
def init():
    hist_obj.set_data([])
    timer.set_text("")
    for i in range(n_particles):
        # lines[i].set_data([], [])
        dots[i].set_data([], [])        
    # return timer, *lines, *dots  
    return hist_obj, timer, *dots # 解包很重要！


# 更新新一帧的数据 refresh
def update(n):
    hist_obj.set_data(data_velocs[n])
    timer.set_text("time = {:.3f}".format(n * dt))
    for i in range(n_particles):
        # lines[i].set_data(data_coords[:n, i, 0], data_coords[:n, i, 1])
        dots[i].set_data(data_coords[n, i, 0], data_coords[n, i, 1])        
    # return timer, *lines, *dots 
    return hist_obj, timer, *dots # 解包很重要！


# perform animation
ani = animation.FuncAnimation(
    fig,
    update,
    frames=np.arange(0, n_increments + 1, 1, dtype=int),
    interval=0,  # 帧之间的延迟（毫秒）默认为200
    blit=True,  # 是否执行blitting优化
    init_func=init,
)


# show animation
plt.show()
