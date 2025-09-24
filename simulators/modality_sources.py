import numpy as np
from scipy.stats import multivariate_normal

class ModalityGenerator:
    def __init__(self, dimensions):
        self.width, self.height, self.depth = dimensions
        self.grid_x, self.grid_y, self.grid_z = np.mgrid[0:self.width, 0:self.height, 0:self.depth]
        self.pos = np.stack((self.grid_x, self.grid_y, self.grid_z), axis=-1)

    def _create_hotspot(self, center, cov_matrix, amplitude):
        """创建一个高斯热点"""
        rv = multivariate_normal(mean=center, cov=cov_matrix)
        return amplitude * rv.pdf(self.pos)

    def generate_crowd_traffic(self, params):
        """生成人群和交通热点"""
        field = np.zeros((self.width, self.height, self.depth))
        for hotspot in params['hotspots']:
            center = hotspot['center']
            # 允许椭球形状
            cov = np.diag([hotspot['scale_x']**2, hotspot['scale_y']**2, hotspot['scale_z']**2])
            field += self._create_hotspot(center, cov, hotspot['amplitude'])
        return np.clip(field, 0, 1)

    def generate_terrain_complexity(self, params):
        """生成地形复杂度，使用Perlin噪声模拟"""
        # 简化的Perlin噪声
        freq_x, freq_y = params['frequency_x'], params['frequency_y']
        octaves, persistence = params['octaves'], params['persistence']
        
        field = np.zeros((self.width, self.height, self.depth))
        amplitude = 1.0
        total_amplitude = 0.0
        
        for _ in range(octaves):
            noise = np.sin(self.grid_x * freq_x + params['seed']) * np.cos(self.grid_y * freq_y + params['seed'])
            field += noise * amplitude
            total_amplitude += amplitude
            amplitude *= persistence
            freq_x *= 2
            freq_y *= 2

        field /= total_amplitude
        field = (field + 1) / 2 # 归一化到 [0, 1]
        field = field * params['amplitude']
        
        # 地形只影响低空
        mask = self.grid_z < self.depth * params['height_ratio']
        return field * mask

    def generate_communication_signal(self, params):
        """生成通信信号强度，基站+遮挡"""
        field = np.zeros((self.width, self.height, self.depth))
        for station in params['stations']:
            dist = np.linalg.norm(self.pos - station['center'], axis=-1)
            # 简单的信号衰减模型
            signal = station['amplitude'] / (1 + (dist / station['radius'])**2)
            field += signal
        
        field = 1.0 - np.clip(field, 0, 1) # 反转，值越大表示信号越差（风险越高）
        return field

    def generate_wind_field(self, params):
        """生成风场，包含梯度风和旋涡"""
        field = np.zeros((self.width, self.height, self.depth))
        
        # 梯度风
        grad_dir = np.array(params['gradient']['direction'])
        grad_dir = grad_dir / np.linalg.norm(grad_dir)
        grad_field = (self.pos @ grad_dir) * params['gradient']['strength']
        field += grad_field

        # 旋涡
        for vortex in params['vortices']:
            center = np.array(vortex['center'])
            dist_xy = np.linalg.norm(self.pos[:,:,:,:2] - center[:2], axis=-1)
            # 兰金涡模型简化
            vortex_field = vortex['strength'] * (1 - np.exp(-(dist_xy**2) / (vortex['radius']**2)))
            field += vortex_field

        # "区间最优"逻辑，风太大或太小都不好
        optimal = params['optimal_speed']
        deviation = np.abs(field - optimal)
        
        # 归一化
        max_dev = np.max(deviation)
        if max_dev > 0:
            deviation /= max_dev
            
        return deviation
