import numpy as np

def calculate_heuristic_risk(modalities_data, weights):
    """
    根据预定义的公共安全规则计算启发式风险值（伪真值）。

    Args:
        modalities_data (np.ndarray): [N, num_modalities] 的多模态数据。
                                      列顺序: 0:crowd, 1:traffic, 2:terrain, 3:communication, 4:wind
        weights (dict): 包含各类风险权重的字典。

    Returns:
        np.ndarray: [N, 1] 的启发式风险值。
    """
    crowd = modalities_data[:, 0]
    traffic = modalities_data[:, 1]
    terrain = modalities_data[:, 2]
    comm_signal_weakness = modalities_data[:, 3] # 模拟器已处理，值越高信号越差
    wind_deviation = modalities_data[:, 4]       # 模拟器已处理，值越高风越不理想

    # 1. 核心公共安全风险：人群和交通的协同效应
    # 使用交互项来体现“协同是核心风险放大因素”
    core_safety_risk = (weights['w_crowd'] * crowd +
                        weights['w_traffic'] * traffic +
                        weights['w_interaction'] * (crowd * traffic))

    # 2. 环境调节因子：飞行条件越“好”，无人机越可能在此区域长时间停留，
    # 因此从公共安全角度看，潜在风险被“放大”。
    # 地形复杂度直接放大风险
    terrain_factor = 1 + weights['w_terrain'] * terrain
    
    # 风扰动小 (deviation -> 0)，因子接近1；风扰动大 (deviation -> 1)，因子减小
    # 这代表飞行条件恶劣时，无人机不会久留，巡检优先级反而下降
    wind_factor = 1 - weights['w_wind'] * wind_deviation
    
    # 通信信号强 (weakness -> 0)，因子接近1；信号弱 (weakness -> 1)，因子减小
    comm_factor = 1 - weights['w_comm'] * comm_signal_weakness
    
    # 最终风险 = 核心风险 * 环境调节因子
    # 确保因子不会为负
    final_risk = core_safety_risk * np.maximum(0, terrain_factor) * np.maximum(0, wind_factor) * np.maximum(0, comm_factor)

    # 归一化到 0-1 范围以便于训练
    max_risk = np.max(final_risk)
    if max_risk > 0:
        final_risk /= max_risk
        
    return final_risk.reshape(-1, 1)
