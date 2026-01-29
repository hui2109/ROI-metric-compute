import numpy as np
from ..interfaces.MetricParam import MetricParam


def nps(metricParam: MetricParam):
    """
        计算 NPS 五个基础指标：
        nps_low / nps_high / nps_ring / nps_total / nps_directional
        所需字段：
            ctx['img']       → 计算图像
            ctx['spacing']   → 像素间距
            ctx['tag'] → 'all' | 'up' | 'down' | 'left' | 'right'
    """

    croped_roi_origianl = metricParam.croped_roi_all.croped_roi_origianl
    PixelSpacing = metricParam.PixelSpacing

    # 1. 确保输入图像是二维HU矩阵
    if croped_roi_origianl.ndim != 2:
        raise ValueError("输入图像必须是二维矩阵")

    # 2. 对图像进行傅里叶变换
    croped_roi_origianl = croped_roi_origianl - np.mean(croped_roi_origianl)  # 去除直流分量
    fft_image = np.fft.fftshift(np.fft.fft2(croped_roi_origianl))

    # 3. 计算功率谱   # 原始二维的NPS
    power_spectrum = np.abs(fft_image) ** 2

    # 4. 计算频率的二维坐标系，中心化
    dx, dy = PixelSpacing[1], PixelSpacing[0]  # 从 CT 图像的 spacing 参数获取像素间距 目的： 乘以 采样率（像素间距的倒数），以转换为实际的 空间频率（单位：mm⁻¹）。
    freqs_x = np.fft.fftfreq(croped_roi_origianl.shape[1], d=dx)  # X方向空间频率 (mm⁻¹)
    freqs_y = np.fft.fftfreq(croped_roi_origianl.shape[0], d=dy)  # Y方向空间频率 (mm⁻¹)
    freqs_x, freqs_y = np.meshgrid(freqs_x, freqs_y)

    # 5. 可选：计算环形平均功率谱
    r = np.sqrt(freqs_x ** 2 + freqs_y ** 2)
    r = np.floor(r * max(croped_roi_origianl.shape))  # 量化为整数索引

    # 计算环形平均
    bins = np.arange(0, np.max(r) + 1)
    nps_, _ = np.histogram(r, bins=bins, weights=power_spectrum)
    counts, _ = np.histogram(r, bins=bins)
    nps = nps_ / counts

    # 6. 计算每个频率点的角度（以度为单位）
    angles = np.arctan2(freqs_y, freqs_x) * 180 / np.pi

    # 7. 计算方向性功率谱  中心化后 0度为x-正方向,计算每个1度范围的方向性噪声  并按照金属伪影的方向 根据不同的mask位置来计算方向性噪声
    angle_range = 1  # 每1度范围

    # 根据不同的方向进行计算
    directional_nps_up = calculate_directional_power(angles, power_spectrum, 0, 180, angle_range)
    directional_nps_down = calculate_directional_power(angles, power_spectrum, -180, 0, angle_range)
    directional_nps_left = calculate_directional_power(angles, power_spectrum, 90, 270, angle_range)
    directional_nps_right = calculate_directional_power(angles, power_spectrum, -90, 90, angle_range)
    directional_nps_360 = calculate_directional_power(angles, power_spectrum, 0, 360, angle_range)

    # 8. 设定频率阈值划分高低频
    f_threshold = np.max(r) / 2  # 设定低频和高频的分界线
    low_freq_mask = r < f_threshold  # 低频部分的掩码
    high_freq_mask = r >= f_threshold  # 高频部分的掩码

    # 计算低频和高频噪声能量
    low_freq_nps = np.sum(power_spectrum[low_freq_mask])
    high_freq_nps = np.sum(power_spectrum[high_freq_mask])

    # 9. 计算能量总和：积分求和
    total_power_spectrum = np.sum(power_spectrum)  # 计算总功率谱的能量
    total_nps_ring = np.sum(nps)  # 计算环形平均功率谱的能量
    total_directional_power_up = np.sum(directional_nps_up)  # 计算方向性功率谱的能量
    total_directional_power_down = np.sum(directional_nps_down)
    total_directional_power_left = np.sum(directional_nps_left)
    total_directional_power_right = np.sum(directional_nps_right)
    total_directional_power_360 = np.sum(directional_nps_360)

    return {'nps':
        {
            'total_power_spectrum': total_power_spectrum,
            'total_nps_ring': total_nps_ring,

            'total_directional_power_up': total_directional_power_up,
            'total_directional_power_down': total_directional_power_down,
            'total_directional_power_left': total_directional_power_left,
            'total_directional_power_right': total_directional_power_right,
            'total_directional_power_360': total_directional_power_360,

            'directional_nps_up': directional_nps_up,
            'directional_nps_down': directional_nps_down,
            'directional_nps_left': directional_nps_left,
            'directional_nps_right': directional_nps_right,
            'directional_nps_360': directional_nps_360,

            'low_freq_nps': low_freq_nps,
            'high_freq_nps': high_freq_nps,
        }
    }


def calculate_directional_power(angles, power_spectrum, start_angle, end_angle, angle_range):
    directional_power = np.zeros((end_angle - start_angle) // angle_range)

    for i in range(directional_power.shape[0]):
        lower_bound = start_angle + i * angle_range
        upper_bound = start_angle + (i + 1) * angle_range
        direction_mask = (angles >= lower_bound) & (angles < upper_bound)
        directional_power[i] = np.sum(power_spectrum[direction_mask])

    return directional_power
