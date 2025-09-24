import numpy as np
import os
from tqdm import tqdm
from simulators.modality_sources import ModalityGenerator
from core.heuristic_risk import calculate_heuristic_risk

# --- 配置 ---
NUM_SAMPLES = 1000  # 您想生成的训练样本数量
DIMS = [50, 50, 10]
OUTPUT_DIR = "training_data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 启发式风险权重 (您可以在此调整) ---
heuristic_weights = {
    'w_crowd': 0.4,
    'w_traffic': 0.3,
    'w_interaction': 0.8,  # 高权重，突出协同效应的重要性
    'w_wind': 0.5,         # 让飞行条件对巡检优先级有显著影响
    'w_comm': 0.4,
    'w_terrain': 0.6       # 让复杂地形成为一个关键的风险放大因素
}

# --- 初始化 ---
modality_gen = ModalityGenerator(DIMS)

# --- 生成循环 ---
for i in tqdm(range(NUM_SAMPLES), desc="Generating training data"):
    # 1. 随机化场景参数
    # 例如，随机生成1到3个热点
    crowd_params = {'hotspots': []}
    for _ in range(np.random.randint(1, 4)):
        crowd_params['hotspots'].append({
            "center": np.random.rand(3) * DIMS,
            "scale_x": np.random.uniform(3, 10),
            "scale_y": np.random.uniform(3, 10),
            "scale_z": np.random.uniform(1, 3),
            "amplitude": np.random.uniform(0.5, 1.0)
        })
    # ... 对 traffic, wind, terrain 等也进行类似的随机化 ...
    # (这里为了简化只展示了crowd的随机化)
    traffic_params = crowd_params # 简化示例
    terrain_params = {"frequency_x": 0.1, "frequency_y": 0.1, "octaves": 3, "persistence": 0.5, "amplitude": 1.0, "height_ratio": 0.3, "seed": np.random.randint(0, 100)}
    comm_params = {"stations": [{"center": [25, 25, 10], "radius": 30, "amplitude": 1.0}]}
    wind_params = {"gradient": {"direction": [0.5, 0.5, 0], "strength": 0.01}, "vortices": [], "optimal_speed": 5.0}


    # 2. 生成多模态数据
    crowd_field = modality_gen.generate_crowd_traffic(crowd_params).flatten()
    traffic_field = modality_gen.generate_crowd_traffic(traffic_params).flatten()
    terrain_field = modality_gen.generate_terrain_complexity(terrain_params).flatten()
    comm_field = modality_gen.generate_communication_signal(comm_params).flatten()
    wind_field = modality_gen.generate_wind_field(wind_params).flatten()

    modalities_data = np.stack([crowd_field, traffic_field, terrain_field, comm_field, wind_field], axis=1)

    # 3. 计算伪真值标签
    heuristic_risk = calculate_heuristic_risk(modalities_data, heuristic_weights)

    # 4. 保存数据和标签
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, f"sample_{i}.npz"),
        data=modalities_data.astype(np.float16), # 使用半精度以节省空间
        label=heuristic_risk.astype(np.float16)
    )

print(f"Successfully generated {NUM_SAMPLES} samples in '{OUTPUT_DIR}'")
