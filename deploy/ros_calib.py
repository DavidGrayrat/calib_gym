import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.spatial.transform import Rotation

## debug调试
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 5678))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
## debug调试

# 关闭所有图形窗口
plt.close('all')

shiftrange = 1
f_ecd = 100
dt_ecd = 1 / f_ecd
num_joint = 3

# 滑窗参数
slidewindow = 8 * shiftrange
starttime = 5
startposition = round(starttime * f_ecd)

# CCA超参数
epsilon_b = 0.9
zeta_b = 0.015
zeta_u = 20

# 读取数据
# w_imu_g_witht = pd.read_csv("extracted_data_100_withimu_imu.txt", delim_whitespace=True, header=None)imu_extracted_001_1
w_imu_g_witht = pd.read_csv("imu_extracted_001_1.txt", sep=",", header=0, skiprows=1)
w_imu_g_witht = w_imu_g_witht.astype(float)
w_imu_g = w_imu_g_witht.iloc[:, 1:4].values

# joint = pd.read_csv("extracted_data_100_withimu_joint.txt", delim_whitespace=True, header=None)
joint = pd.read_csv("state_extracted_001_1.txt", sep=",", header=0, skiprows=1)
joint = joint.astype(float)
w_joint_witht = joint.iloc[:, [0, 2, 4, 6]].values
theta_joint_witht = joint.iloc[:, [0, 1, 3, 5]].values
w_joint = w_joint_witht[:, 1:4]
theta_joint = theta_joint_witht[:, 1:4]

w_joint1 = w_joint[:, 0]
w_joint2 = w_joint[:, 1]
w_joint3 = w_joint[:, 2]

# 计算末端执行器角速度
w_effector = np.column_stack((
    w_joint1 * np.sin(theta_joint[:, 1]) * np.sin(theta_joint[:, 2]) -
    w_joint1 * np.cos(theta_joint[:, 1]) * np.sin(theta_joint[:, 2]) -
    w_joint1 * np.sin(theta_joint[:, 1]) * np.cos(theta_joint[:, 2]) -
    w_joint1 * np.cos(theta_joint[:, 1]) * np.cos(theta_joint[:, 2]),
    
    w_joint1 * np.cos(theta_joint[:, 1]) * np.sin(theta_joint[:, 2]) -
    w_joint1 * np.cos(theta_joint[:, 1]) * np.cos(theta_joint[:, 2]) +
    w_joint1 * np.sin(theta_joint[:, 1]) * np.cos(theta_joint[:, 2]) +
    w_joint1 * np.sin(theta_joint[:, 1]) * np.sin(theta_joint[:, 2]),
    
    w_joint2 + w_joint3
))

w_imu_i = w_effector

# CCA分析
G = w_imu_g[startposition:startposition + int(slidewindow * f_ecd) + 1, :]
Gx = G[:, 0] - np.mean(G[:, 0])
Gy = G[:, 1] - np.mean(G[:, 1])
Gz = G[:, 2] - np.mean(G[:, 2])

sigma_gg = np.array([
    [np.mean(Gx * Gx), np.mean(Gx * Gy), np.mean(Gx * Gz)],
    [np.mean(Gy * Gx), np.mean(Gy * Gy), np.mean(Gy * Gz)],
    [np.mean(Gz * Gx), np.mean(Gz * Gy), np.mean(Gz * Gz)]
])

r_ig_td = np.zeros((2 * shiftrange * f_ecd + 1, 
                    2 * shiftrange * f_ecd + 1, 
                    2 * shiftrange * f_ecd + 1))

theta_joint_read = np.zeros((int(slidewindow * f_ecd) + 1, num_joint))
w_joint_window = np.zeros((int(slidewindow * f_ecd) + 1, num_joint))

# 时间偏移估计
print("开始计算时间偏移...")
total_iterations = (2 * shiftrange * f_ecd + 1) ** 3
current_iter = 0

for i, td_ecd_1 in enumerate(np.linspace(-shiftrange, shiftrange, 2 * shiftrange * f_ecd + 1)):
    print(f"Processing td_ecd_1: {td_ecd_1:.2f}")
    pd_ecd_1 = int(round(td_ecd_1 * f_ecd))
    
    for j, td_ecd_2 in enumerate(np.linspace(-shiftrange, shiftrange, 2 * shiftrange * f_ecd + 1)):
        pd_ecd_2 = int(round(td_ecd_2 * f_ecd))
        
        for k, td_ecd_3 in enumerate(np.linspace(-shiftrange, shiftrange, 2 * shiftrange * f_ecd + 1)):
            pd_ecd_3 = int(round(td_ecd_3 * f_ecd))
            
            # 读取编码器数据
            idx1 = startposition + pd_ecd_1
            idx2 = startposition + pd_ecd_2
            idx3 = startposition + pd_ecd_3
            end = int(slidewindow * f_ecd)
            
            theta_joint_read[:, 0] = theta_joint[idx1:idx1 + end + 1, 0]
            theta_joint_read[:, 1] = theta_joint[idx2:idx2 + end + 1, 1]
            theta_joint_read[:, 2] = theta_joint[idx3:idx3 + end + 1, 2]
            
            w_joint_window[:, 0] = w_joint[idx1:idx1 + end + 1, 0]
            w_joint_window[:, 1] = w_joint[idx2:idx2 + end + 1, 1]
            w_joint_window[:, 2] = w_joint[idx3:idx3 + end + 1, 2]
            
            # 计算假想IMU读数
            I = np.column_stack((
                w_joint_window[:, 0] * np.sin(theta_joint_read[:, 1]) * np.sin(theta_joint_read[:, 2]) -
                w_joint_window[:, 0] * np.cos(theta_joint_read[:, 1]) * np.sin(theta_joint_read[:, 2]) -
                w_joint_window[:, 0] * np.sin(theta_joint_read[:, 1]) * np.cos(theta_joint_read[:, 2]) -
                w_joint_window[:, 0] * np.cos(theta_joint_read[:, 1]) * np.cos(theta_joint_read[:, 2]),
                
                w_joint_window[:, 0] * np.cos(theta_joint_read[:, 1]) * np.sin(theta_joint_read[:, 2]) -
                w_joint_window[:, 0] * np.cos(theta_joint_read[:, 1]) * np.cos(theta_joint_read[:, 2]) +
                w_joint_window[:, 0] * np.sin(theta_joint_read[:, 1]) * np.cos(theta_joint_read[:, 2]) +
                w_joint_window[:, 0] * np.sin(theta_joint_read[:, 1]) * np.sin(theta_joint_read[:, 2]),
                
                w_joint_window[:, 1] + w_joint_window[:, 2]
            ))
            
            Ix = I[:, 0] - np.mean(I[:, 0])
            Iy = I[:, 1] - np.mean(I[:, 1])
            Iz = I[:, 2] - np.mean(I[:, 2])
            
            sigma_gi = np.array([
                [np.mean(Gx * Ix), np.mean(Gx * Iy), np.mean(Gx * Iz)],
                [np.mean(Gy * Ix), np.mean(Gy * Iy), np.mean(Gy * Iz)],
                [np.mean(Gz * Ix), np.mean(Gz * Iy), np.mean(Gz * Iz)]
            ])
            
            sigma_ii = np.array([
                [np.mean(Ix * Ix), np.mean(Ix * Iy), np.mean(Ix * Iz)],
                [np.mean(Iy * Ix), np.mean(Iy * Iy), np.mean(Iy * Iz)],
                [np.mean(Iz * Ix), np.mean(Iz * Iy), np.mean(Iz * Iz)]
            ])
            
            sigma_ig = sigma_gi.T
            
            try:
                inv_gg = np.linalg.inv(sigma_gg)
                inv_ii = np.linalg.inv(sigma_ii)
                trace_val = np.trace(inv_gg @ sigma_gi @ inv_ii @ sigma_ig)
                r_ig_td[i, j, k] = np.sqrt((1/3) * trace_val)
            except np.linalg.LinAlgError:
                r_ig_td[i, j, k] = 0

# 寻找最大值
r_ig_td_max = np.max(r_ig_td)
max_indices = np.unravel_index(np.argmax(r_ig_td), r_ig_td.shape)

ts_ecd_1_estimate = (max_indices[0] - shiftrange * f_ecd) / f_ecd
ts_ecd_2_estimate = (max_indices[1] - shiftrange * f_ecd) / f_ecd
ts_ecd_3_estimate = (max_indices[2] - shiftrange * f_ecd) / f_ecd

# 绘制结果
axis_x = np.linspace(-shiftrange, shiftrange, 2 * shiftrange * f_ecd + 1)
axis_y = np.lin(axis_x)  # 仅示例，实际应选择合适的轴数据

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(axis_x, axis_y)
ax.plot_surface(X, Y, r_ig_td[:, :, max_indices[2]], cmap='viridis')
ax.scatter(axis_x[max_indices[1]], axis_y[max_indices[0]], r_ig_td_max, color='r')
ax.set_xlabel('td_ecd_2')
ax.set_ylabel('td_ecd_1')
ax.set_zlabel('r_ig_td')
plt.show()

# 旋转估计
ps_ecd_1_estimate = int(round(ts_ecd_1_estimate * f_ecd))
ps_ecd_2_estimate = int(round(ts_ecd_2_estimate * f_ecd))
ps_ecd_3_estimate = int(round(ts_ecd_3_estimate * f_ecd))

theta_joint_correct = np.zeros((int(slidewindow * f_ecd) + 1, num_joint))
theta_joint_correct[:, 0] = theta_joint[startposition + ps_ecd_1_estimate : startposition + ps_ecd_1_estimate + int(slidewindow * f_ecd) + 1, 0]
theta_joint_correct[:, 1] = theta_joint[startposition + ps_ecd_2_estimate : startposition + ps_ecd_2_estimate + int(slidewindow * f_ecd) + 1, 1]
theta_joint_correct[:, 2] = theta_joint[startposition + ps_ecd_3_estimate : startposition + ps_ecd_3_estimate + int(slidewindow * f_ecd) + 1, 2]

w_joint_correct = np.zeros((int(slidewindow * f_ecd) + 1, num_joint))
w_joint_correct[:, 0] = w_joint[startposition + ps_ecd_1_estimate : startposition + ps_ecd_1_estimate + int(slidewindow * f_ecd) + 1, 0]
w_joint_correct[:, 1] = w_joint[startposition + ps_ecd_2_estimate : startposition + ps_ecd_2_estimate + int(slidewindow * f_ecd) + 1, 1]
w_joint_correct[:, 2] = w_joint[startposition + ps_ecd_3_estimate : startposition + ps_ecd_3_estimate + int(slidewindow * f_ecd) + 1, 2]

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

Ix_correct = I_correct[:, 0] - np.mean(I_correct[:, 0])
Iy_correct = I_correct[:, 1] - np.mean(I_correct[:, 1])
Iz_correct = I_correct[:, 2] - np.mean(I_correct[:, 2])

sigma_ii_correct = np.array([
    [np.mean(Ix_correct**2), np.mean(Ix_correct*Iy_correct), np.mean(Ix_correct*Iz_correct)],
    [np.mean(Iy_correct*Ix_correct), np.mean(Iy_correct**2), np.mean(Iy_correct*Iz_correct)],
    [np.mean(Iz_correct*Ix_correct), np.mean(Iz_correct*Iy_correct), np.mean(Iz_correct**2)]
])

sigma_ig_correct = np.array([
    [np.mean(Ix_correct*Gx), np.mean(Ix_correct*Gy), np.mean(Ix_correct*Gz)],
    [np.mean(Iy_correct*Gx), np.mean(Iy_correct*Gy), np.mean(Iy_correct*Gz)],
    [np.mean(Iz_correct*Gx), np.mean(Iz_correct*Gy), np.mean(Iz_correct*Gz)]
])

U, S, Vh = svd(np.linalg.inv(sigma_ii_correct) @ sigma_ig_correct)
V = Vh.T
R_gi_estimate = (U @ np.diag([1, 1, np.linalg.det(U @ V.T)]) @ V.T).T

# 将旋转矩阵转换为欧拉角
rot = Rotation.from_matrix(R_gi_estimate)
theta_ig_estimate = np.round(rot.as_euler('xyz', degrees=True), 2) + np.array([0, 360, 0])

cond_num_sigma_ii = np.linalg.cond(sigma_ii_correct)
lambda_min = np.min(np.abs(np.linalg.eigvals(sigma_ii_correct)))

print("sigma_ii =")
print(sigma_ii_correct)

if r_ig_td_max > epsilon_b and lambda_min > zeta_b and cond_num_sigma_ii < zeta_u:
    print("满足可观测性条件")
else:
    print("不满足可观测性条件")

print(f"r_ig_td_max={r_ig_td_max}, epsilon_b={epsilon_b}")
print(f"abs_lambda_min={lambda_min}, zeta_b={zeta_b}")
print(f"sigma_ii矩阵条件数={cond_num_sigma_ii}, zeta_u={zeta_u}")

print(f"编码器1时移估计：{ts_ecd_1_estimate}, 估计误差：{1000*ts_ecd_1_estimate}ms")
print(f"编码器2时移估计：{ts_ecd_2_estimate}, 估计误差：{1000*ts_ecd_2_estimate}ms")
print(f"编码器3时移估计：{ts_ecd_3_estimate}, 估计误差：{1000*ts_ecd_3_estimate}ms")

print("旋转矩阵估计：")
print(R_gi_estimate)
print("旋转角度估计：", theta_ig_estimate)