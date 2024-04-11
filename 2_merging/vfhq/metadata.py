from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel


__all__ = ["EBinPose", "Pose", "SimpleBoundingBox", "BoundingBox", "FacePrediction"]


class EBinPose(str, Enum):
    FRONTAL = "frontal"
    PROFILE_LEFT = "profile_left"
    PROFILE_RIGHT = "profile_right"
    PROFILE_UP = "profile_up"
    PROFILE_DOWN = "profile_down"
    PROFILE_EXTREME = "profile_extreme"


class Pose(BaseModel):
    pitch: Optional[float] = None
    roll: Optional[float] = None
    yaw: Optional[float] = None
    bin: Optional[EBinPose] = None


class SimpleBoundingBox(BaseModel):
    ltrb: list
    pose: Pose

    @staticmethod
    def parse_from_file(filepath: Path):
        res = []
        with open(filepath) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        for line in lines:
            splits = line.split(" ")
            ltrb = [float(x) for x in splits]
            pose = Pose()
            pose.yaw = float(splits[4])
            pose.pitch = float(splits[5])
            pose.roll = float(splits[6])
            res.append(SimpleBoundingBox(ltrb=ltrb, pose=pose))
        return res


class BoundingBox(BaseModel):
    ltrb_gt: Optional[list]
    ltrb_directmhp: Optional[list]
    ltrb_retina: Optional[list]
    lmks5: Optional[list]
    lmks68: Optional[list]
    pose_directmhp: Optional[Pose]
    pose_synergynet: Optional[Pose]
    pose_lmk68: Optional[Pose]

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def get_nparray_from_txt(textsplits, startidx, endidx=None):
        if endidx is None:
            endidx = len(textsplits)
        return np.array([float(x) for x in textsplits[startidx:endidx]])


class FacePrediction(BaseModel):
    frame_id: str
    clip_id: str
    boundingbox: Optional[BoundingBox]

    @staticmethod
    def parse_vfhq_annotation(txtline, clip_id):
        res = FacePrediction.parse_obj({"frame_id": "", "clip_id": "", "boundingbox": None})
        res.boundingbox = BoundingBox()
        # FRAME INDEX X0 Y0 W H [Landmarks (5 Points)]
        splits = txtline.split(" ")
        res.frame_id = splits[0]
        res.clip_id = clip_id
        ltwh = BoundingBox.get_nparray_from_txt(splits, 2, 6)
        ltbr = np.array([ltwh[0], ltwh[1], ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]])
        res.boundingbox.ltrb_gt = ltbr.tolist()
        if len(splits) > 6:
            res.boundingbox.lmks5 = BoundingBox.get_nparray_from_txt(splits, 6).tolist()
        return res
