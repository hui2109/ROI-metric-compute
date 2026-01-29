import numpy as np

from .load_data import LoadData
from collections import Counter, OrderedDict
from pprint import pprint
from pandas import DataFrame
from .constant import METRICS
from .ROI import get_rectangle_roi_dict
from .interfaces import MetricParam
from .interfaces import CropedROI
from typing import Callable, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from typing import Union


def get_roi_info(data: LoadData):
    num_patients = len(data.exams)
    roi_name_list = []

    for rTStruct in data.rTStructs.values():
        for roiRegion in rTStruct.StructureSetROISequence:
            roi_name_list.append(roiRegion.ROIName)

    c = Counter(roi_name_list)
    roi_name_set = sorted(list(c.keys()), key=lambda x: c[x], reverse=True)

    roi_name_dict = {roi_name: [None] * num_patients for roi_name in roi_name_set}
    df_dict = {
        'index': [None] * num_patients,
        'dir_name': [None] * num_patients,
        **roi_name_dict
    }
    for row_idx, (dirName, rtStruct) in enumerate(data.rTStructs.items()):
        for roiName in roi_name_set:
            df_dict['index'][row_idx] = row_idx
            df_dict['dir_name'][row_idx] = dirName

            for roiRegion in rtStruct.StructureSetROISequence:
                if roiName == roiRegion.ROIName:
                    df_dict[roiName][row_idx] = 1

    df = DataFrame.from_dict(df_dict)
    return df


def compute_spcROI_spcMETRIC(data: LoadData, *, roi_name_list: list[str], metric_func_list: list[METRICS]):
    exams = list(data.exams.values())
    image_Volumes, mask_Volumes = __get_target_Volumes(data, roi_name_list)

    workers = os.cpu_count() or 1
    futures = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for idx1_patient, mask_Volume_all in enumerate(mask_Volumes):
            if len(mask_Volume_all) == 0:
                continue

            pixel_spacing = exams[idx1_patient].CT[0].Images[0].PixelSpacing

            for idx2_roi, mask_Volume in enumerate(mask_Volume_all):
                rectangle_roi_dict = get_rectangle_roi_dict(image_Volumes[idx1_patient], mask_Volume)

                for idx3_img, (croped_roi_origianl, croped_roi_mask, croped_roi_target) in rectangle_roi_dict.items():
                    metric_param = MetricParam(
                        croped_roi_all=CropedROI(
                            croped_roi_origianl=croped_roi_origianl,
                            croped_roi_mask=croped_roi_mask,
                            croped_roi_target=croped_roi_target,
                        ),
                        PixelSpacing=pixel_spacing,
                    )

                    info = [idx1_patient, roi_name_list[idx2_roi], idx3_img]

                    # 一个任务算完所有 metrics（减少 submit 数量）
                    futures.append(
                        executor.submit(__compute_many_metrics, metric_param, metric_func_list, info)
                    )

        print(f"Submitted {len(futures)} tasks.")

    results = {}
    for future in as_completed(futures):
        results.update(future.result())

    return dict(
        sorted(
            results.items(),
            key=lambda item: __parse_key(item[0])
        )
    )


def __parse_key(key: str):
    idx_patient, roi_name, idx_img, metric_name = key.split("_")
    return int(idx_patient), roi_name, int(idx_img), metric_name


def __compute_many_metrics(
        metric_param: MetricParam,
        metric_func_list: list[Callable[[MetricParam], Any]],
        info: tuple[int, str, int],
):
    idx_patient, roi_name, idx_img = info
    out = {}

    # ⚠️ 多进程里频繁 print 很慢，建议关掉或限流
    # print(f"patient_idx={idx_patient}, roi_name={roi_name}, slice_idx={idx_img}")

    for metric_func in metric_func_list:
        key = f"{idx_patient}_{roi_name}_{idx_img}_{metric_func.__name__}"
        out[key] = metric_func(metric_param)

    return out


def __get_target_Volumes(data: LoadData, roi_name_list: list[str]):
    exams = list(data.exams.values())
    rtstructs = list(data.rTStructs.values())

    image_Volumes = []
    mask_Volumes = []

    for idx, rtstruct in enumerate(rtstructs):
        mask_Volume_1 = []

        for roi_name in roi_name_list:
            for roiRegion in rtstruct.StructureSetROISequence:
                if roi_name == roiRegion.ROIName:
                    mask_Volume_1.append(roiRegion.Volume)

        mask_Volumes.append(mask_Volume_1)
        if mask_Volume_1:
            image_Volumes.append(exams[idx].CT[0].Volume)
        else:
            image_Volumes.append(mask_Volume_1)

    return image_Volumes, mask_Volumes
