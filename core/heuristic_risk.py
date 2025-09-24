import numpy as np

def calculate_heuristic_risk(modalities_data, weights):
    """
    根据强化的、多层次的公共安全规则计算启发式风险值（伪真值）。
    这个函数是训练模型的“教师”，其定义的规则将通过训练被GDIF-Net模型学习。

    Args:
        modalities_data (np.ndarray): [N, num_modalities] 的多模态数据。
                                      列顺序: 0:crowd, 1:traffic, 2:terrain, 3:communication, 4:wind
        weights (dict): 包含各类风险权重的字典，用于调整规则的强度。

    Returns:
        np.ndarray: [N, 1] 的启发式风险值，已归一化。
    """
    # 从输入数据中解析各个模态
    crowd = modalities_data[:, 0]
    traffic = modalities_data[:, 1]
    terrain_complexity = modalities_data[:, 2]
    comm_signal_weakness = modalities_data[:, 3]  # 值越高信号越差
    wind_deviation = modalities_data[:, 4]        # 值越高风越不理想

    # --- 规则 1 & 2: 核心风险与协同放大 ---
    # 规则1: 基础风险由人群和交通线性决定
    risk_base = weights['w_crowd'] * crowd + weights['w_traffic'] * traffic
    # 规则2: 人车混行的协同效应，通过乘法项实现风险放大
    risk_synergy = weights['w_interaction'] * (crowd * traffic)
    # 核心公共安全风险 = 基础风险 + 协同风险
    core_risk = risk_base + risk_synergy

    # --- 规则 3: 有利飞行环境的“风险吸引”调节 ---
    # 将不利因素（风扰、信号弱）转化为一个值域在[0,1]区间的“有利度”调节因子
    # 因子值越接近1，代表飞行条件越好，巡检价值越高
    wind_factor = 1.0 - weights['w_wind'] * wind_deviation
    comm_factor = 1.0 - weights['w_comm'] * comm_signal_weakness
    # 确保调节因子不会为负
    environmental_modulation_factor = np.maximum(0.01, wind_factor) * np.maximum(0.01, comm_factor)

    # --- 规则 4: 地形复杂度的直接风险增强 ---
    # 将地形复杂度转化为一个值域在[1, 1+w]区间的风险放大因子
    terrain_amplification_factor = 1.0 + weights['w_terrain'] * terrain_complexity

    # --- 最终风险计算：整合所有规则 ---
    # 最终风险 = 核心风险 * 环境调节因子 * 地形放大因子
    final_risk = core_risk * environmental_modulation_factor * terrain_amplification_factor
    
    # 将最终风险归一化到 0-1 范围，以便于模型训练
    max_risk = np.max(final_risk)
    if max_risk > 0:
        final_risk /= max_risk
        
    return final_risk.reshape(-1, 1)
