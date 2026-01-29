import numpy as np
import scipy.ndimage as ndi
from ..interfaces.MetricParam import MetricParam


def diameter(metricParam: MetricParam):
    croped_roi_origianl, croped_roi_mask = metricParam.croped_roi_all.croped_roi_origianl, metricParam.croped_roi_all.croped_roi_mask
    PixelSpacing = metricParam.PixelSpacing
    thresholds = metricParam.ThresholdsForDiameter

    res = {}

    # 提前检查是否为3个连通区域
    labeled, n_cc = ndi.label(croped_roi_mask)

    if n_cc != 3:
        return res  # 非 3 管结构，跳过

    # 计算连通区域的质心，提取左右两管
    cc_props = ndi.center_of_mass(croped_roi_mask, labeled, range(1, n_cc + 1))
    cc_sorted = sorted(enumerate(cc_props, start=1), key=lambda x: x[1][1])  # 按X坐标排序
    labels_to_use = [cc_sorted[0][0], cc_sorted[2][0]]  # 最左和最右两个label

    for th in thresholds:
        d_row, d_col = cal_diameter(
            croped_roi_origianl,
            croped_roi_mask,
            PixelSpacing,
            th,
            labeled,
            labels_to_use,
        )
        res[f'row_d_{th}'] = round(d_row, 2)
        res[f'col_d_{th}'] = round(d_col, 2)
    return {'diameter': res}


def cal_diameter(croped_roi_origianl, croped_roi_mask, PixelSpacing, th, labeled, labels_to_use):
    """
        识别掩膜中的三个施源器截面（连通域），仅取最左与次左两管，
        在中心 3×3 像素块内统计 HU>th 的像素并转 mm。
        返回 dict: row_mm, col_mm, row_counts, col_counts
    """

    row_spacing, col_spacing = PixelSpacing

    # 获取掩膜mask的xy最大范围
    ys, xs = np.where(croped_roi_mask > 0)  # 找到非零像素的行、列索引
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    row_counts, col_counts = [], []

    for lab, i in zip(labels_to_use, [1, 2, 3]):
        ys, xs = np.where(labeled == lab)
        # 获取当前连通区域的范围
        ys_min, ys_max = ys.min(), ys.max()
        xs_min, xs_max = xs.min(), xs.max()
        y_c = int(np.round(np.mean(ys)))
        x_c = int(np.round(np.mean(xs)))

        rows3 = [y for y in (y_c - 1, y_c, y_c + 1) if y_min <= y <= y_max]
        cols3 = [x for x in (x_c - 1, x_c, x_c + 1) if x_min <= x <= x_max]

        # 行向统计（对选行求均值）
        if len(labels_to_use) == 2 and i == 1:  # 代表遍历的是第一个 也就是最左边的连通区域 这里要注意计算的列范围为最左边到连通区域右边一定范围。 否则可能会不小心计算到下一个连通区域内的高像素值
            pix_row = [np.sum(croped_roi_origianl[y, :xs_max + 2] > th) for y in rows3]
        elif len(labels_to_use) == 2 and i == 2:  # 代表是最右边的连通区域
            pix_row = [np.sum(croped_roi_origianl[y, xs_min - 2:] > th) for y in rows3]
        elif len(labels_to_use) == 1 and i == 1:  # 代表估计是单管层面 不存在三管 那么就计算每一行的左右两侧所有
            pix_row = [np.sum(croped_roi_origianl[y, :] > th) for y in rows3]
        else:
            raise ValueError(f"❌ 检测到{len(labels_to_use)} 连通域，当前遍历第{i}.")

        row_counts.append(np.mean(pix_row))

        # 列向统计（对选列求均值）
        pix_col = [np.sum(croped_roi_origianl[:, x] > th) for x in cols3]
        col_counts.append(np.mean(pix_col))

    # 平均像素个数
    row_counts = np.array(row_counts)
    col_counts = np.array(col_counts)

    # 行列平均物理长度
    row_mm = np.mean(row_counts) * col_spacing
    col_mm = np.mean(col_counts) * row_spacing

    return row_mm, col_mm
