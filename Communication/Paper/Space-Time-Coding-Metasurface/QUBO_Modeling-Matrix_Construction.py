import time
import numpy as np
import csv
# --------------------------参数设置（对应论文）--------------------------
# 空间与时间维度参数
M, N = 8, 8  # 超表面空间行数、列数（m:0~M-1, n:0~N-1）
L = 4  # 时间编码序列长度（l:0~L-1）
n_bit = 2  # 相位编码比特数（1或2）
h = 0  # 目标谐波阶数（h=0对应中心频率）
# 波束方向参数
theta0, phi0 = np.deg2rad(-14.5), 0  # 主波束方向（θ, φ）
theta_main_low = theta0 - 0.1  # 主波束区域下界（弧度）
theta_main_high = theta0 + 0.1  # 主波束区域上界（弧度）
theta_side = np.linspace(-np.pi / 2, np.pi / 2, 20)  # 旁瓣离散角度（积分近似）
# 权重参数（平衡主波束与旁瓣）
w_main = 10.0  # 主波束增强权重
w_side = -1.0  # 旁瓣抑制权重（负值表示惩罚）
# 物理参数（归一化）
lambda0 = 1.0  # 波长
k0 = 2 * np.pi / lambda0  # 自由空间波数
d = lambda0 / 2  # 超原子间距（λ/2，满足Nyquist采样）
# 2比特编码系数（满足|c1|² + |c2|² = 1
c = [1.0 / np.sqrt(2), np.exp(1.0j*np.pi/2)/ np.sqrt(2)]
# 总超原子数与总变量数计算
total_atoms = M * N * L  # 空间×时间超原子总数
total_spins = total_atoms * n_bit  # QUBO变量总数（每个超原子n_bit个变量）
print(f"总变量数：{total_spins}（需≤550以适配CIM）")
start_time = time.time()  # 记录开始时间

# --------------------------1. 索引解析函数--------------------------
def parse_index(p):
    """
    将QUBO变量索引p解析为物理参数：
    输出：(m, n, l, b)
        m: 空间行索引（0~M-1）
        n: 空间列索引（0~N-1）
        l: 时间槽索引（0~L-1）
        b: 比特索引（0~n_bit-1）
    索引规则：p = atom_idx * n_bit + b，其中atom_idx = m*(N*L) + n*L + l
    """
    atom_idx = p // n_bit  # 超原子索引（剥离比特维度）
    b = p % n_bit  # 比特索引（0~n_bit-1）
    # 解析超原子的空间-时间索引（atom_idx = m*(N*L) + n*L + l）
    l = atom_idx % L
    rem_after_l = atom_idx // L  # 剩余部分：m*N + n
    n = rem_after_l % N
    m = rem_after_l // N

    return m, n, l, b
# --------------------------2. 耦合系数A计算--------------------------
def compute_A(m_p, n_p, l_p, m_q, n_q, l_q, theta, phi):
    """计算特定方向(theta, phi)的耦合系数A_pq^h（论文式52）"""
    # 波数分量
    kx = k0 * np.sin(theta) * np.cos(phi)
    ky = k0 * np.sin(theta) * np.sin(phi)
    # 空间相位差
    spatial_phase = kx * (m_p - m_q) * d + ky * (n_p - n_q) * d
    # 时间相位差
    temporal_phase = -2 * np.pi * h * (l_p - l_q) / L
    # Sinc调制因子
    if h == 0:
        sinc_sq = 1.0  # sinc(0) = 1
    else:
        sinc_val = np.sin(np.pi * h / L)
        sinc_sq = (sinc_val / (np.pi * h / L)) ** 2
    # 单个超原子的远场方向图（论文假设为余弦分布）
    E_sq = np.cos(theta)
    A = (E_sq**2 / (L ** 2)) * sinc_sq * np.exp(1j * (spatial_phase + temporal_phase))
    return A


# --------------------------3. 物理耦合系数J矩阵计算（含主波束+旁瓣）--------------------------
print("步骤1/3：计算物理耦合系数J矩阵...")
# 初始化J矩阵
J_phys = np.zeros((total_spins, total_spins), dtype=np.float64)

for p in range(total_spins):
    m_p, n_p, l_p, b_p = parse_index(p)

    for q in range(p, total_spins):  # 利用对称性，仅计算上三角
        m_q, n_q, l_q, b_q = parse_index(q)

        # 3.1 计算主波束耦合系数A_main
        A_main = compute_A(m_p, n_p, l_p, m_q, n_q, l_q, theta0, phi0)

        # 3.2 计算旁瓣区域平均耦合系数A_side（积分近似）
        A_side_sum = 0.0
        side_count = 0
        for ths in theta_side:
            if not (theta_main_low < ths < theta_main_high):  # 排除主波束
                A_ths = compute_A(m_p, n_p, l_p, m_q, n_q, l_q, ths, 0)
                A_side_sum += A_ths
                side_count += 1
        A_side = A_side_sum / side_count if side_count > 0 else 0.0

        # 3.3 总耦合系数A_total = 主波束项 + 旁瓣项
        A_total = w_main * A_main + w_side * A_side

        # 3.4 计算自旋模型耦合系数J（区分1比特/2比特）
        if n_bit == 1:
            # 1比特：J_pq = Re(A_total)
            J_val = np.real(A_total)
        else:
            # 2比特：J_ab^pq = Re(c[b_p] * conj(c[b_q]) * A_total)
            J_val = np.real(c[b_p] * np.conj(c[b_q]) * A_total)

        # 3.5 对称赋值（J_phys[p,q] = J_phys[q,p]）
        J_phys[p, q] = J_val
        if p != q:
            J_phys[q, p] = J_val

# --------------------------4. QUBO矩阵计算（严格应用映射公式）--------------------------
print("步骤2/3：将J矩阵转换为QUBO矩阵...")
qubo = np.zeros((total_spins, total_spins), dtype=np.float64)

for p in range(total_spins):
    # 4.1 计算线性项系数a_p（对角项）
    # 公式：a_p = 4 * sum(J_phys[p,q] for q in all if q != p)
    sum_j = 0.0
    for q in range(total_spins):
        if q != p:
            sum_j += J_phys[p, q]
    a_p = 4 * sum_j  # 线性项系数（1比特/2比特通用）
    qubo[p, p] = a_p  # 对角项存储线性项

    # 4.2 计算二次项系数b_pq（非对角项，p < q）
    for q in range(p + 1, total_spins):  # 仅计算上三角非对角项
        # 公式：b_pq = -8 * J_phys[p,q]（1比特/2比特通用）
        b_pq = -8 * J_phys[p, q]
        qubo[p, q] = b_pq
        qubo[q, p] = b_pq  # 对称赋值

# --------------------------5. 矩阵缩放与导出（适配硬件要求）--------------------------
print("步骤3/3：缩放矩阵并导出...")
# 手动缩放矩阵以适配int8范围
max_abs_val = np.max(np.abs(qubo))
if max_abs_val > 0:
    scale_factor = 127.0 / max_abs_val
    print(f"应用缩放因子: {scale_factor:.6f} (最大绝对值: {max_abs_val:.4f})")
    qubo_scaled = np.round(qubo * scale_factor).astype(np.int8)
else:
    qubo_scaled = np.zeros_like(qubo, dtype=np.int8)

with open(
        'qubo_matrix_complete.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for row in qubo_scaled:
        writer.writerow(row)


end_time = time.time()
total_time = end_time - start_time
minutes = int(total_time // 60)
seconds = total_time % 60
# 验证输出信息
print("\nQUBO矩阵生成完成，参数验证：")
print(f"- 矩阵维度：{qubo_scaled.shape[0]}×{qubo_scaled.shape[1]}")
print(f"- 数值范围：[{np.min(qubo_scaled)}, {np.max(qubo_scaled)}]")
print(f"- 数据类型：{qubo_scaled.dtype}")
print(f"- 非零元素占比：{np.count_nonzero(qubo_scaled) / (total_spins ** 2):.2%}")
print(f"- 导出路径：qubo_matrix_complete.csv")
print(f"- 主程序执行时间：{minutes}分{seconds:.2f}秒")