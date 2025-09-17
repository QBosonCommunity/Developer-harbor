import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

matplotlib.rcParams["font.family"] = ["SimHei"]
matplotlib.rcParams['axes.unicode_minus'] = False

# --------------------------参数设置（与建模代码一致）--------------------------
# 空间与时间维度参数
M, N = 8, 8  # 超表面空间行数、列数
L = 4  # 时间编码序列长度
n_bit = 2  # 相位编码比特数（1或2）
h = 0  # 目标谐波阶数

# 波束方向参数
theta0, phi0 = np.deg2rad(-14.5), 0  # 主波束目标方向
target_theta_deg = np.rad2deg(theta0)  # 目标角度（度）

# 物理参数
lambda0 = 1.0  # 波长
k0 = 2 * np.pi / lambda0  # 自由空间波数
d = lambda0 / 2  # 超原子间距（λ/2）

# 2比特编码系数（1bit系数为1，不显示）
c = [1.0 / np.sqrt(2), np.exp(1.0j*np.pi/2) / np.sqrt(2)]

# 总变量数（与建模代码一致）
total_atoms = M * N * L
total_spins = total_atoms * n_bit  # QUBO变量总数


# --------------------------1. 索引解析函数--------------------------
def parse_index(p):
    """与建模代码parse_index完全一致，确保索引映射统一"""
    atom_idx = p // n_bit  # 超原子索引（剥离比特维度）
    b = p % n_bit  # 比特索引（0~n_bit-1）

    # 解析超原子的空间-时间索引（atom_idx = m*(N*L) + n*L + l）
    l = atom_idx % L
    rem_after_l = atom_idx // L
    n = rem_after_l % N
    m = rem_after_l // N

    return m, n, l, b


# --------------------------2. 相位解码函数--------------------------
def decode_solution(qubo_solution):
    """根据建模代码的编码规则解码相位"""
    phase = np.zeros((M, N, L), dtype=np.complex128)
    for p in range(total_spins):
        m, n, l, b = parse_index(p)
        x = qubo_solution[p]
        s = 2 * x - 1  # 自旋变量映射（0→-1，1→+1）
        phase[m, n, l] += c[b] * s
    phase /= np.abs(phase)
    return phase


# --------------------------3. 远场功率计算--------------------------
def calc_far_field(phase, h=0):
    """根据建模代码的耦合系数A计算逻辑，计算远场功率"""
    theta_range = np.linspace(-np.pi / 2, np.pi / 2, 720)  # 角度范围：-90°到90°
    P = np.zeros_like(theta_range)

    for idx, theta in enumerate(theta_range):
        # 波数分量
        kx = k0 * np.sin(theta) * np.cos(phi0)
        ky = k0 * np.sin(theta) * np.sin(phi0)
        E_total = 0j

        for m in range(M):
            for n in range(N):
                for l in range(L):
                    # 空间相位
                    spatial_phase = kx * m * d + ky * n * d
                    # 时间相位
                    temporal_phase = -2 * np.pi * h * l / L
                    E_total += phase[m, n, l] * np.exp(1j * (spatial_phase + temporal_phase))

        # 功率计算
        if h == 0:
            sinc_sq = 1.0
        else:
            sinc_val = np.sin(np.pi * h / L)
            sinc_sq = (sinc_val / (np.pi * h / L)) ** 2
        # 远场方向图
        E_sq = np.cos(theta) ** 2
        P[idx] = np.abs(E_total) ** 2 * sinc_sq * E_sq / (L ** 2)

    return theta_range, P

if __name__ == "__main__":
    # 输入最优解（QUBO解）
    qubo_solution = np.array(
        [ 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
          1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
          0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
          0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
          0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
          1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
          1, 0, 1, 0, 1, 0, 1, 0 ]
    )
    # 修正解长度
    if len(qubo_solution) < total_spins:
        qubo_solution = np.pad(qubo_solution, (0, total_spins - len(qubo_solution)), mode='constant')
    elif len(qubo_solution) > total_spins:
        qubo_solution = qubo_solution[:total_spins]
    print(f"QUBO解长度：{len(qubo_solution)}（应等于{total_spins}）")

    # 解码相位
    phase = decode_solution(qubo_solution)
    print(f"相位解码完成，维度：(M, N, L,n-bit) = {(M, N, L,n_bit)}")

    # 相位离散值验证
    phase_angles = np.angle(phase) % (2 * np.pi)  # 归一化到[0, 2π)
    phase_normalized = np.round(phase_angles / np.pi, 2)
    unique_phases = np.unique(phase_normalized)
    print(f"相位离散值验证：{unique_phases}π rad")

    # 计算远场功率
    theta_range, P_raw = calc_far_field(phase, h=h)
    theta_deg = np.rad2deg(theta_range)
    max_power = np.max(P_raw)  # 以主瓣功率为基准归一化
    P = P_raw / max_power
    P_dB = 10 * np.log10(P + 1e-10)

    # 4.1 主瓣定位（全局最大功率点）
    peak_idx = np.argmax(P_raw)
    mainlobe_theta = theta_deg[peak_idx]
    main_power = P[peak_idx]  # 应为1.0

    # 4.2 旁瓣抑制比（SLL）
    main_theta_rad = theta_range[peak_idx]
    # 排除主瓣±5°范围（与建模代码的主波束区域逻辑一致）
    sidelobe_mask = np.abs(theta_range - main_theta_rad) > np.deg2rad(40)
    sidelobe_power = np.max(P[sidelobe_mask]) if np.any(sidelobe_mask) else 0
    sll = 10 * np.log10(sidelobe_power / main_power) if main_power > 0 else -np.inf

    # 4.3 角度误差（与建模目标角度对比）
    angle_error = np.abs(mainlobe_theta - target_theta_deg)

    # 4.4 3dB波束宽度
    peak_power_dB = P_dB[peak_idx]
    half_power_dB = peak_power_dB - 3
    # 左侧3dB点
    left_mask = (theta_range < main_theta_rad) & (P_dB <= half_power_dB)
    left_idx = np.where(left_mask)[0]
    left_3db = theta_deg[left_idx[-1]] if len(left_idx) > 0 else theta_deg[0]
    # 右侧3dB点
    right_mask = (theta_range > main_theta_rad) & (P_dB <= half_power_dB)
    right_idx = np.where(right_mask)[0]
    right_3db = theta_deg[right_idx[0]] if len(right_idx) > 0 else theta_deg[-1]
    beamwidth_3db = right_3db - left_3db

    # 打印评估结果
    print("\n=== 性能评估结果 ===")
    print(f"主瓣功率（归一化）：{main_power:.2f}")
    print(f"旁瓣最大值（归一化）：{sidelobe_power:.4f}")
    print(f"旁瓣抑制比（SLL）：{sll:.2f} dB")
    print(f"主瓣角度(°)：{mainlobe_theta:.2f}")
    print(f"目标角度(°)：{target_theta_deg:.2f}")
    print(f"角度误差(°)：{angle_error:.2f}")
    print(f"3dB波束宽度(°)：{beamwidth_3db:.2f}")

    # 4.5 谐波抑制评估（与建模代码的h=0目标对比）
    harmonics = [-2, -1, 1, 2]
    harmonic_suppression = []
    for h_harm in harmonics:
        _, P_harm_raw = calc_far_field(phase, h=h_harm)
        # 提取谐波在主瓣角度处的功率
        harm_power = P_harm_raw[peak_idx]
        # 抑制比 = 10*log10(谐波功率 / 主瓣功率)
        suppression = 10 * np.log10((harm_power / max_power) + 1e-10)
        harmonic_suppression.append(suppression)

    print("\n=== 非目标谐波抑制（dB） ===")
    for h_harm, supp in zip(harmonics, harmonic_suppression):
        print(f"h={h_harm}: {supp:.2f} dB")

    # 可视化
    fig = plt.figure(figsize=(11, 5))

    # 极坐标图
    ax1 = plt.subplot(121, polar=True)
    ax1.plot(theta_range, P, 'b-', linewidth=2)
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_ylim(0, 1.1)
    ax1.set_title('远场功率方向图（极坐标）')
    ax1.plot([theta0, theta0], [0, 1], 'r--', label='目标方向')
    ax1.plot([theta_range[peak_idx], theta_range[peak_idx]], [0, 1], 'g-', label='主瓣方向')
    ax1.legend(loc='lower right')

    # 直角坐标图
    ax2 = plt.subplot(122)
    ax2.plot(theta_deg, P_dB, 'b-', linewidth=2)
    ax2.axvline(x=target_theta_deg, color='red', linestyle='--', label=f'目标角度: {target_theta_deg:.1f}°')
    ax2.axvline(x=mainlobe_theta, color='green', linestyle='-', label=f'主瓣角度: {mainlobe_theta:.1f}°')
    ax2.axhline(y=sll, color='orange', linestyle=':', label=f'旁瓣抑制比: {sll:.1f}dB')
    ax2.axvspan(left_3db, right_3db, color='gray', alpha=0.2, label=f'3dB宽度: {beamwidth_3db:.1f}°')

    ax2.set_xlabel('θ (°)')
    ax2.set_ylabel('归一化功率 (dB)')
    ax2.set_title('远场功率方向图（直角坐标）')
    ax2.set_xlim(-90, 90)
    ax2.set_ylim(-50, 5)
    ax2.xaxis.set_major_locator(MultipleLocator(30))
    ax2.xaxis.set_minor_locator(MultipleLocator(10))
    ax2.grid(which='both', linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.show()