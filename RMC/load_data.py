from pathlib import Path
import pydicom
import pickle

from .interfaces.OneExam import OneImage, OneCTSeries, OneExam
from .interfaces.RTDose import RTDose
from .interfaces.RTPlan import RTPlan
from .interfaces.RTStruct import Contour, ROIRegion, RTStruct
from .ROI.gen_mask import gen_roi_volume
from datetime import date
import numpy as np


class LoadData:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.exams: dict[str, OneExam] = {}
        self.rTDoses: dict[str, RTDose] = {}
        self.rTPlans: dict[str, RTPlan] = {}
        self.rTStructs: dict[str, RTStruct] = {}

        self.load_data()

    def load_data(self):
        if Path(self.data_dir / 'my_saved_data').exists():
            with open(self.data_dir / 'my_saved_data' / 'Exams.pkl', 'rb') as f:
                self.exams = pickle.load(f)
            with open(self.data_dir / 'my_saved_data' / 'RTDoses.pkl', 'rb') as f:
                self.rTDoses = pickle.load(f)
            with open(self.data_dir / 'my_saved_data' / 'RTPlans.pkl', 'rb') as f:
                self.rTPlans = pickle.load(f)
            with open(self.data_dir / 'my_saved_data' / 'RTStructs.pkl', 'rb') as f:
                self.rTStructs = pickle.load(f)
            return

        self.sorted_dcm_files = {}

        for subdir in self.data_dir.iterdir():
            CT_dcm_files = []
            MRI_dcm_files = []
            RD_dcm_files = []
            RP_dcm_files = []
            RS_dcm_files = []

            for dcm_file in subdir.glob("*.dcm"):
                with pydicom.dcmread(dcm_file) as ds:
                    if ds.Modality == 'CT':
                        CT_dcm_files.append(ds)
                    elif ds.Modality == 'MRI':
                        MRI_dcm_files.append(ds)
                    elif ds.Modality == 'RTDOSE':
                        RD_dcm_files.append(ds)
                    elif ds.Modality == 'RTPLAN':
                        RP_dcm_files.append(ds)
                    elif ds.Modality == 'RTSTRUCT':
                        RS_dcm_files.append(ds)
                    else:
                        print(f'Modality not recognized! {dcm_file}, {ds.Modality}.')

            self.sorted_dcm_files['CT_dcm_files'] = CT_dcm_files
            self.sorted_dcm_files['MRI_dcm_files'] = MRI_dcm_files
            self.sorted_dcm_files['RD_dcm_files'] = RD_dcm_files
            self.sorted_dcm_files['RP_dcm_files'] = RP_dcm_files
            self.sorted_dcm_files['RS_dcm_files'] = RS_dcm_files

            oneExam, rTDose, rTPlan, rTStruct = self.__init_data()
            self.exams[subdir.name] = oneExam
            self.rTDoses[subdir.name] = rTDose
            self.rTPlans[subdir.name] = rTPlan
            self.rTStructs[subdir.name] = rTStruct

        self.__save_data()  # 保存二进制数据

    def __init_data(self):
        saved_oneExam = {}

        # 处理CT数据
        savedCTSeries = {}
        for ds in self.sorted_dcm_files['CT_dcm_files']:
            oneImage = OneImage(
                SOPInstanceUID=ds.SOPInstanceUID,
                InstanceNumber=ds.InstanceNumber,
                Rows=ds.Rows,
                Columns=ds.Columns,
                PixelSpacing=ds.PixelSpacing,
                ImagePositionPatient=ds.ImagePositionPatient,
                BitsAllocated=ds.BitsAllocated,
                pixel_array=ds.pixel_array,
                RescaleSlope=ds.RescaleSlope,
                RescaleIntercept=ds.RescaleIntercept,
                WindowWidth=ds.WindowWidth,
                WindowCenter=ds.WindowCenter,
                # ds_object=ds  先不写, 数据太大了
            )

            if savedCTSeries.get(ds.SeriesInstanceUID) is None:
                savedCTSeries[ds.SeriesInstanceUID] = OneCTSeries(
                    Modality=ds.Modality,
                    SeriesInstanceUID=ds.SeriesInstanceUID,
                    SeriesNumber=str(ds.SeriesNumber),
                    SeriesDate=date(int(ds.SeriesDate[:4]), int(ds.SeriesDate[4:6]), int(ds.SeriesDate[6:])),
                    ImageOrientationPatient=ds.ImageOrientationPatient,
                    SliceThickness=ds.SliceThickness,
                    SliceLocation=ds.SliceLocation,
                    KVP=ds.KVP,
                    XRayTubeCurrent=ds.XRayTubeCurrent,
                    ConvolutionKernel=ds.ConvolutionKernel,
                    Manufacturer=ds.Manufacturer + ' | ' + ds.ManufacturerModelName,
                    Images=[oneImage],
                )
            else:
                savedCTSeries[ds.SeriesInstanceUID].Images.append(oneImage)

            if not saved_oneExam:
                saved_oneExam['PatientID'] = ds.PatientID
                saved_oneExam['given_name'] = ds.PatientName.given_name
                saved_oneExam['PatientSex'] = ds.PatientSex
                saved_oneExam['StudyDate'] = date(int(ds.StudyDate[:4]), int(ds.StudyDate[4:6]), int(ds.StudyDate[6:]))
                saved_oneExam['StudyInstanceUID'] = ds.StudyInstanceUID

        # 处理MRI数据
        savedMRISeries = {}
        for ds in self.sorted_dcm_files['MRI_dcm_files']:
            pass

        # 处理RD数据
        ds = self.sorted_dcm_files['RD_dcm_files'][0]
        rTDose = RTDose(
            Modality=ds.Modality,
            DoseUnits=ds.DoseUnits,
            # ds_object=ds,
        )

        # 处理RP数据
        ds = self.sorted_dcm_files['RP_dcm_files'][0]
        rTPlan = RTPlan(
            Modality=ds.Modality,
            RTPlanLabel=ds.RTPlanLabel,
            # ds_object=ds,
        )

        # 处理RS数据
        ds = self.sorted_dcm_files['RS_dcm_files'][0]
        roiRegions = []
        for idx, roi in enumerate(ds.ROIContourSequence):
            contours = []

            for con in getattr(roi, 'ContourSequence', []):
                contour = Contour(
                    ContourData=con.ContourData,
                    ContourGeometricType=con.ContourGeometricType,
                    NumberOfContourPoints=con.NumberOfContourPoints,
                    ReferencedSOPClassUIDs=[getattr(i, 'ReferencedSOPClassUID', None)
                                            for i in getattr(con, 'ContourImageSequence', {})],
                    ReferencedSOPInstanceUIDs=[getattr(i, 'ReferencedSOPInstanceUID', None)
                                               for i in getattr(con, 'ContourImageSequence', {})],
                )
                contours.append(contour)

            if contours:
                roiRegion = ROIRegion(
                    ROIName=ds.StructureSetROISequence[idx].ROIName,
                    ROINumber=ds.StructureSetROISequence[idx].ROINumber,
                    ROIGenerationAlgorithm=ds.StructureSetROISequence[idx].ROIGenerationAlgorithm,
                    ReferencedFrameOfReferenceUID=ds.StructureSetROISequence[idx].ReferencedFrameOfReferenceUID,
                    ROIDisplayColor=roi.ROIDisplayColor,
                    RTROIInterpretedType=ds.RTROIObservationsSequence[idx].RTROIInterpretedType,
                    Contours=contours,
                )
                roiRegions.append(roiRegion)

        rTStruct = RTStruct(
            Modality=ds.Modality,
            SOPClassUID=ds.SOPClassUID,
            SOPInstanceUID=ds.SOPInstanceUID,
            FrameOfReferenceUID=ds.FrameOfReferenceUID,
            ImageSeriesInstanceUID=ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID,
            StructureSetLabel=ds.StructureSetLabel,
            StructureSetROISequence=roiRegions,
            # ds_object=ds,
        )

        # oneExam数据整合
        # 图像顺序整理
        for ct_series in savedCTSeries.values():
            ct_series.Images.sort(key=lambda x: x.InstanceNumber)
        for mri_series in savedMRISeries.values():
            mri_series.Images.sort(key=lambda x: x.InstanceNumber)

        saved_oneExam['CT'] = list(savedCTSeries.values())
        saved_oneExam['MRI'] = list(savedMRISeries.values())
        oneExam = OneExam(**saved_oneExam)

        # 计算CT和MRI的3D volume
        for ct_series in oneExam.CT:
            ct_series.Volume = np.stack([img.pixel_array * img.RescaleSlope + img.RescaleIntercept for img in ct_series.Images], axis=0)
        for mri_series in oneExam.MRI:
            mri_series.Volume = np.stack([img.pixel_array * img.RescaleSlope + img.RescaleIntercept for img in mri_series.Images], axis=0)

        # 计算每一个ROIRegion的Volume
        paired_CT_series = savedCTSeries[rTStruct.ImageSeriesInstanceUID]
        for roi_region in rTStruct.StructureSetROISequence:
            roi_region.Volume = gen_roi_volume(paired_CT_series, roi_region)

        return oneExam, rTDose, rTPlan, rTStruct

    def __save_data(self):
        if not Path(self.data_dir / 'my_saved_data').is_dir():
            Path(self.data_dir / 'my_saved_data').mkdir(parents=True)

        with open(self.data_dir / 'my_saved_data' / 'Exams.pkl', 'wb') as f:
            pickle.dump(self.exams, f)
        with open(self.data_dir / 'my_saved_data' / 'RTDoses.pkl', 'wb') as f:
            pickle.dump(self.rTDoses, f)
        with open(self.data_dir / 'my_saved_data' / 'RTPlans.pkl', 'wb') as f:
            pickle.dump(self.rTPlans, f)
        with open(self.data_dir / 'my_saved_data' / 'RTStructs.pkl', 'wb') as f:
            pickle.dump(self.rTStructs, f)
