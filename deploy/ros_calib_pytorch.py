import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.spatial.transform import Rotation
import torch
import time

# 配置PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
# torch.set_float32_matmul_precision('high')  # 启用TF32加速

# 参数设置
shiftrange = 1
f_ecd = 100
dt_ecd = 1 / f_ecd
num_joint = 3
slidewindow = 8 * shiftrange
starttime = 5
startposition = round(starttime * f_ecd)
epsilon_b = 0.9
zeta_b = 0.015
zeta_u = 20

# 自定义CSV读取函数
def load_csv_data(filename, cols, skiprows=0, delimiter=','):
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for _ in range(skiprows):
            next(reader)
        for row in reader:
            data.append([float(row[i]) for i in cols])
    return np.array(data)

# 加载IMU数据 (格式: 时间戳, wx, wy, wz)
w_imu_g = load_csv_data("imu_extracted_001_1.txt", cols=[1,2,3], skiprows=1)

# 加载关节状态数据 (格式: 时间戳, theta1, w1, theta2, w2, theta3, w3)
joint_data = load_csv_data("state_extracted_001_1.txt", 
                         cols=[1,3,5],  # 选择theta1, theta2, theta3
                         skiprows=1)
theta_joint = joint_data[:, 0:3]
w_joint = joint_data[:, 3:6]

# 转换为PyTorch张量并移至GPU
def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32, device=device)

w_imu_g_tensor = to_tensor(w_imu_g)
theta_joint_tensor = to_tensor(theta_joint)
w_joint_tensor = to_tensor(w_joint)

# 预计算G矩阵相关
G_tensor = w_imu_g_tensor[startposition:startposition + int(slidewindow*f_ecd)+1]
G_centered = G_tensor - G_tensor.mean(dim=0)
sigma_gg = (G_centered.T @ G_centered) / (G_tensor.size(0) - 1)
inv_gg = torch.linalg.pinv(sigma_gg)

# 生成所有可能的时间偏移组合
shift_steps = 2*shiftrange*f_ecd + 1
offsets = torch.linspace(-shiftrange, shiftrange, shift_steps, device=device)
pd_values = (offsets * f_ecd).round().long()

# 主计算函数 (GPU加速)
def compute_correlations():
    window_size = int(slidewindow*f_ecd) + 1
    base_indices = torch.arange(window_size, device=device)
    
    # 预分配结果张量
    r_ig_td = torch.zeros((shift_steps, shift_steps, shift_steps), device=device)
    
    # 批量处理第一个维度
    for i in range(shift_steps):
        start = time.time()
        
        # 获取关节1的窗口数据
        idx1 = startposition + pd_values[i]
        valid_length = min(len(w_joint_tensor)-idx1, window_size)
        pad_length = max(window_size - valid_length, 0)
        
        # 处理数据边界
        w0 = w_joint_tensor[idx1:idx1+valid_length, 0]
        theta1 = theta_joint_tensor[idx1:idx1+valid_length, 1]
        theta2 = theta_joint_tensor[idx1:idx1+valid_length, 2]
        
        # 补零处理
        if pad_length > 0:
            w0 = torch.cat([w0, torch.zeros(pad_length, device=device)])
            theta1 = torch.cat([theta1, torch.zeros(pad_length, device=device)])
            theta2 = torch.cat([theta2, torch.zeros(pad_length, device=device)])
        
        # 预计算三角函数
        sin_t1 = torch.sin(theta1)
        cos_t1 = torch.cos(theta1)
        sin_t2 = torch.sin(theta2)
        cos_t2 = torch.cos(theta2)
        
        # 计算I矩阵的前两列
        I0 = w0 * (sin_t1*sin_t2 - cos_t1*sin_t2 - sin_t1*cos_t2 - cos_t1*cos_t2)
        I1 = w0 * (cos_t1*sin_t2 - cos_t1*cos_t2 + sin_t1*cos_t2 + sin_t1*sin_t2)
        
        # 批量处理第二、三维度
        for j in range(shift_steps):
            idx2 = startposition + pd_values[j]
            w1 = w_joint_tensor[idx2:idx2+window_size, 1]
            for k in range(shift_steps):
                idx3 = startposition + pd_values[k]
                w2 = w_joint_tensor[idx3:idx3+window_size, 2]
                
                # 构建I矩阵
                I2 = w1 + w2
                I = torch.stack([I0, I1, I2], dim=1)
                I_centered = I - I.mean(dim=0)
                
                # 计算协方差矩阵
                sigma_gi = (G_centered.T @ I_centered) / (G_tensor.size(0)-1)
                sigma_ii = (I_centered.T @ I_centered) / (I_centered.size(0)-1)
                
                # 计算相关系数
                try:
                    inv_ii = torch.linalg.pinv(sigma_ii)
                    trace_val = torch.trace(inv_gg @ sigma_gi @ inv_ii @ sigma_gi.T)
                    r_ig_td[i,j,k] = torch.sqrt((1/3)*trace_val)
                except:
                    r_ig_td[i,j,k] = 0
        
        print(f"Processed {i+1}/{shift_steps} | Time: {time.time()-start:.2f}s")
    
    return r_ig_td.cpu().numpy()

# 执行主计算
print("开始计算时间偏移...")
r_ig_td_np = compute_correlations()

# 寻找最大值
max_idx = np.unravel_index(np.argmax(r_ig_td_np), r_ig_td_np.shape)
ts_estimates = [(pd_values[i].item()/f_ecd) - shiftrange for i in max_idx]

# 后续处理（旋转估计等）与原始代码相同
# ... [此处应包含后续的旋转矩阵计算和可视化代码] ...

# 寻找最大值并转换时间偏移
max_idx = np.unravel_index(np.argmax(r_ig_td_np), r_ig_td_np.shape)
ts_estimates = [(pd_values[i].item()/f_ecd) - shiftrange for i in max_idx]

# 转换为numpy数组用于后续处理
theta_joint_np = theta_joint_tensor.cpu().numpy()
w_joint_np = w_joint_tensor.cpu().numpy()

# 获取校正后的关节数据
def get_corrected_data(index, joint_data):
    pd = pd_values[index].item()
    start = startposition + pd
    end = start + int(slidewindow * f_ecd) + 1
    return joint_data[start:end]

theta_joint_correct = np.column_stack([
    get_corrected_data(max_idx[0], theta_joint_np[:, 0]),
    get_corrected_data(max_idx[1], theta_joint_np[:, 1]),
    get_corrected_data(max_idx[2], theta_joint_np[:, 2])
])

w_joint_correct = np.column_stack([
    get_corrected_data(max_idx[0], w_joint_np[:, 0]),
    get_corrected_data(max_idx[1], w_joint_np[:, 1]),
    get_corrected_data(max_idx[2], w_joint_np[:, 2])
])

# 计算校正后的I矩阵
I_correct = np.column_stack((
    w_joint_correct[:, 0] * np.sin(theta_joint_correct[:, 1]) * np.sin(theta_joint_correct[:, 2]) -
    w_joint_correct[:, 0] * np.cos(theta_joint_correct[:, 1]) * np.sin(theta_joint_correct[:, 2]) -
    w_joint_correct[:, 0] * np.sin(theta_joint_correct[:, 1]) * np.cos(theta_joint_correct[:, 2]) -
    w_joint_correct[:, 0] * np.cos(theta_joint_correct[:, 1]) * np.cos(theta_joint_correct[:, 2]),
    
    w_joint_correct[:, 0] * np.cos(theta_joint_correct[:, 1]) * np.sin(theta_joint_correct[:, 2]) -
    w_joint_correct[:, 0] * np.cos(theta_joint_correct[:, 1]) * np.cos(theta_joint_correct[:, 2]) +
    w_joint_correct[:, 0] * np.sin(theta_joint_correct[:, 1]) * np.cos(theta_joint_correct[:, 2]) +
    w_joint_correct[:, 0] * np.sin(theta_joint_correct[:, 1]) * np.sin(theta_joint_correct[:, 2]),
    
    w_joint_correct[:, 1] + w_joint_correct[:, 2]
))

# 计算协方差矩阵
I_centered = I_correct - np.mean(I_correct, axis=0)
G_centered_np = G_centered.cpu().numpy()

sigma_ii_correct = np.cov(I_centered, rowvar=False)
sigma_ig_correct = np.cov(I_centered, G_centered_np, rowvar=False)[:3, 3:6]

# SVD分解计算旋转矩阵
U, S, Vh = svd(np.linalg.inv(sigma_ii_correct) @ sigma_ig_correct)
V = Vh.T
det_factor = np.linalg.det(U @ V.T)
R_gi_estimate = U @ np.diag([1, 1, det_factor]) @ V.T

# 转换为欧拉角
rot = Rotation.from_matrix(R_gi_estimate)
theta_ig_estimate = np.round(rot.as_euler('xyz', degrees=True) + np.array([0, 360, 0]), 2)

# 计算可观测性条件
cond_num_sigma_ii = np.linalg.cond(sigma_ii_correct)
lambda_min = np.min(np.abs(np.linalg.eigvals(sigma_ii_correct)))

# 可视化结果
axis_x = np.linspace(-shiftrange, shiftrange, 2 * shiftrange * f_ecd + 1)
selected_z = max_idx[2]

fig = plt.figure(figsize=(15, 10))

# 3D 曲面图
ax1 = fig.add_subplot(121, projection='3d')
X, Y = np.meshgrid(axis_x, axis_x)
surf = ax1.plot_surface(X, Y, r_ig_td_np[:, :, selected_z], cmap='viridis', alpha=0.8)
ax1.scatter(axis_x[max_idx[1]], axis_x[max_idx[0]], r_ig_td_np.max(), 
           color='r', s=100, label='Peak Value')
ax1.set_xlabel('td_ecd_2 (s)')
ax1.set_ylabel('td_ecd_1 (s)')
ax1.set_zlabel('Correlation Coefficient')
ax1.set_title('Time Shift Correlation Matrix')
fig.colorbar(surf, ax=ax1, shrink=0.5)

# 条件指标显示
ax2 = fig.add_subplot(122)
conditions = [
    f"Max Correlation: {r_ig_td_np.max():.4f} (Threshold: {epsilon_b})",
    f"Min Eigenvalue: {lambda_min:.4f} (Threshold: {zeta_b})",
    f"Condition Number: {cond_num_sigma_ii:.2f} (Threshold: {zeta_u})"
]
ax2.text(0.1, 0.8, "\n".join(conditions), fontsize=12, 
        bbox=dict(facecolor='whitesmoke', alpha=0.5))
ax2.axis('off')

plt.tight_layout()
plt.show()

# 输出结果
print("\n" + "="*60)
print("最终校准结果：")
print(f"编码器1时移估计: {ts_estimates[0]:.4f}s (误差: {ts_estimates[0]*1000:.1f}ms)")
print(f"编码器2时移估计: {ts_estimates[1]:.4f}s (误差: {ts_estimates[1]*1000:.1f}ms)")
print(f"编码器3时移估计: {ts_estimates[2]:.4f}s (误差: {ts_estimates[2]*1000:.1f}ms)")

print("\n旋转矩阵估计：")
print(np.round(R_gi_estimate, 4))

print("\n欧拉角估计(度)：")
print(f"X轴旋转: {theta_ig_estimate[0]:.2f}°")
print(f"Y轴旋转: {theta_ig_estimate[1]:.2f}°")
print(f"Z轴旋转: {theta_ig_estimate[2]:.2f}°")

observability_condition = (
    r_ig_td_np.max() > epsilon_b and 
    lambda_min > zeta_b and 
    cond_num_sigma_ii < zeta_u
)
print(f"\n可观测性条件 {'满足' if observability_condition else '不满足'}")
print("="*60)
