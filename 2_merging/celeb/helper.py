import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from cython_bbox import bbox_overlaps as bbox_ious
from rich import print


__all__ = [
    "ious",
    "read_csv_from_txt",
    "merge3posedf",
    "mergetxt",
    "bin_a_pose",
    "get_pose_from_row",
]


def ious(atlbrs, btlbrs):
    """Compute cost based on IoU.

    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float),
    )

    return ious


def read_csv_from_txt(txt_path: Path, skiprows=7, skipfooter=1, datatype=None):
    colnames16 = ["frameid", "idx", "x1", "y1", "x2", "y2"] + [
        f"lmk{i//2}_{'x' if i%2==0 else 'y'}" for i in range(10)
    ]
    colnames19 = colnames16 + [
        "synergy_head_pitch",
        "synergy_head_yaw",
        "synergy_head_roll",
    ]
    colnames223 = colnames16.copy() + [
        "poseanh_head_pitch",
        "poseanh_head_yaw",
        "poseanh_head_roll",
    ]
    for i in range(68):
        colnames223.extend([f"3dlmk{i}_x", f"3dlmk{i}_y", f"3dlmk{i}_z"])
    if datatype == "gt":
        header = colnames16
    elif datatype == "poseanh":
        header = colnames19
    elif datatype == "synergy":
        header = colnames223
    else:
        print(datatype)

    df = pd.read_csv(
        txt_path.as_posix(),
        skiprows=skiprows,
        delim_whitespace=True,
        names=header,
        skipfooter=skipfooter,
    )
    # if len(df.columns) == 16:
    #     finalcol = colnames16
    # elif len(df.columns) == 19:
    #     finalcol = colnames19
    # elif len(df.columns) == 223:
    #     finalcol = colnames223
    # df.rename(columns=dict(zip(df.columns, finalcol)), inplace=True)
    return df


def merge3posedf(celeb, synergy, poseanh):
    merged = pd.merge(
        celeb,
        poseanh,
        on=[
            "frameid",
            "idx",
            "x1",
            "x2",
            "y1",
            "y2",
            *[f"lmk{i}_x" for i in range(5)],
            *[f"lmk{i}_y" for i in range(5)],
        ],
        how="inner",
    )
    merged = pd.merge(
        synergy,
        merged,
        on=[
            "frameid",
            "idx",
            "x1",
            "x2",
            "y1",
            "y2",
            *[f"lmk{i}_x" for i in range(5)],
            *[f"lmk{i}_y" for i in range(5)],
        ],
        how="inner",
    )
    return merged


def parseline_celeb(line):  # Retina prediction file
    NCOLS = 16
    numbers = line.split()
    numbers[2:] = list(map(lambda x: round(float(x), 2), numbers[2:]))
    frameid, idx, x1, y1, x2, y2 = numbers[:6]

    if len(numbers) == NCOLS:
        lmks = numbers[6:]
    else:
        lmks = None
    d = dict(frameid=frameid, idx=idx, x1=x1, y1=y1, x2=x2, y2=y2, lmks5pts=lmks)
    return d


def parseline_synergy(line):  # Done
    NCOLS = 223
    numbers = line.split()
    numbers[2:] = list(map(lambda x: round(float(x), 2), numbers[2:]))
    frameid, idx, x1, y1, x2, y2 = numbers[:6]
    pose, lmks68pts = None, None
    if len(numbers) == NCOLS:
        pose = numbers[16:19]
        lmks68pts = numbers[19:]

    d = dict(
        frameid=frameid,
        idx=idx,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        synergy_yaw=pose[0] if pose else None,
        synergy_pitch=pose[1] if pose else None,
        synergy_roll=pose[2] if pose else None,
        lmks68pts=lmks68pts if lmks68pts else None,
    )
    return d


def parseline_poseanh(line):  # Done
    NCOLS = 19
    numbers = line.split()
    numbers[2:] = list(map(lambda x: round(float(x), 2), numbers[2:]))
    frameid, idx, x1, y1, x2, y2 = numbers[:6]
    if len(numbers) == NCOLS:
        pose = numbers[16:19]
    else:
        pose = None

    d = dict(
        frameid=frameid,
        idx=idx,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        poseanh_yaw=pose[0] if pose else None,
        poseanh_pitch=pose[1] if pose else None,
        poseanh_roll=pose[2] if pose else None,
    )
    return d


def parseline_iqa(line):
    numbers = line.split()
    numbers[2:] = list(map(lambda x: round(float(x), 2), numbers[2:]))
    frameid, idx, x1, y1, x2, y2 = numbers[:6]
    iqa = numbers[-1] if len(numbers) > 6 else None
    d = dict(frameid=frameid, idx=idx, x1=x1, y1=y1, x2=x2, y2=y2, iqa=iqa)
    return d


def merge_dict(dict1, dict2):
    res = dict1.copy()
    for k, v in dict2.items():
        if k in res:
            assert res[k] == dict2[k], f"{k}, {res[k]}, {dict2[k]}"
        else:
            res[k] = v
    return res


def mergetxt(gttxt, synergytxt, poseanhtxt, iqatxt):
    with open(gttxt) as f:
        gtlines = f.readlines()[1:]
    with open(synergytxt) as f:
        synergylines = f.readlines()[1:]
    with open(poseanhtxt) as f:
        poseanhlines = f.readlines()[1:]
    with open(iqatxt) as f:
        iqalines = f.readlines()[1:]
    assert len(gtlines) == len(synergylines) == len(poseanhlines) == len(iqalines)
    res = []
    for gtline, synergyline, poseanhline, iqaline in zip(gtlines, synergylines, poseanhlines, iqalines):
        gtline = gtline.strip()
        synergyline = synergyline.strip()
        poseanhline = poseanhline.strip()
        iqaline = iqaline.strip()

        gtdet = parseline_celeb(gtline)
        synergydet = parseline_synergy(synergyline)
        poseanhdet = parseline_poseanh(poseanhline)
        iqadet = parseline_iqa(iqaline)

        finaldet = merge_dict(gtdet, synergydet)
        finaldet = merge_dict(finaldet, poseanhdet)
        finaldet = merge_dict(finaldet, iqadet)
        res.append(finaldet)
    df = pd.DataFrame(res)
    return df


def bin_a_pose(yaw, pitch, roll):
    if math.isnan(yaw) or math.isnan(pitch) or math.isnan(roll) or yaw is None or pitch is None or roll is None:
        return None
    if abs(yaw) < 45 and abs(pitch) < 30:
        bin = "frontal"
    elif abs(yaw) > 90 and abs(pitch) > 90:
        bin = "profile_extreme"
    elif yaw > 45 and yaw < 180 and abs(pitch) < 30:
        bin = "profile_left"
    elif yaw < -45 and yaw > -180 and abs(pitch) < 30:
        bin = "profile_right"
    elif abs(yaw) < 45 and pitch > 30 and pitch < 180:
        bin = "profile_up"
    elif abs(yaw) < 45 and pitch < -30 and pitch > -180:
        bin = "profile_down"
    else:
        bin = "profile_extreme"
    return bin


def get_pose_from_row(row):
    d = defaultdict(dict)
    for p in ["yaw", "pitch", "roll"]:
        for m in ["mhp_", "synergy_", "poseanh_"]:
            d[p][f"{m}{p}"] = row[f"{m}{p}"]
    return d
