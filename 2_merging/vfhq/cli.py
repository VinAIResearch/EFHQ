import concurrent.futures
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from helper import *
from metadata import *
from natsort import natsorted
from pandarallel import pandarallel
from rich import print
from tqdm import tqdm


app = typer.Typer()


@app.command()
def vfhq_posemerge_multithread(
    synergy_path: Path = typer.Argument(..., help="synergy path", exists=True, dir_okay=True),
    poseanh_path: Path = typer.Argument(..., help="poseanh path", exists=True, dir_okay=True),
    iqa_path: Path = typer.Argument(..., help="iqa path", exists=True, dir_okay=True),
    gt_path: Path = typer.Argument(..., help="gt path", exists=True, dir_okay=True),
    output_path: Path = typer.Argument(..., help="output path"),
    workers: int = typer.Option(8, help="nworkers"),
):
    synergytxts = synergy_path.glob("*.txt")
    poseanhtxts = poseanh_path.glob("*.txt")
    iqatxts = iqa_path.glob("*.txt")
    gttxts = gt_path.glob("*.txt")

    synergynames = {x.stem for x in synergytxts}
    poseanhnames = {x.stem for x in poseanhtxts}
    iqanames = {x.stem for x in iqatxts}
    gtnames = {x.stem for x in gttxts}

    overlapnames = synergynames & poseanhnames & gtnames & iqanames
    missingnames = gtnames - overlapnames
    print(f"Total clips matched: {len(overlapnames)}")
    print(f"Total clips unmatched comparing to GT: {len(missingnames)}")

    missing_outpath = output_path / "missing.txt"
    output_path.mkdir(parents=True, exist_ok=True)
    with open(missing_outpath.as_posix(), "w") as f:
        for missingname in missingnames:
            f.write(missingname + "\n")

    def func(overlapname):
        synergytxtpath = synergy_path / (overlapname + ".txt")
        poseanhtxtpath = poseanh_path / (overlapname + ".txt")
        gttxtpath = gt_path / (overlapname + ".txt")
        iqatxtpath = iqa_path / (overlapname + ".txt")
        outputdfpath = output_path / (overlapname + ".csv")

        df = mergetxt(gttxtpath, synergytxtpath, poseanhtxtpath, iqatxtpath)
        df.to_csv(
            outputdfpath,
            index=False,
            columns=[
                "frameid",
                "idx",
                "x1",
                "y1",
                "x2",
                "y2",
                "iqa",
                "synergy_yaw",
                "synergy_pitch",
                "synergy_roll",
                "poseanh_yaw",
                "poseanh_pitch",
                "poseanh_roll",
                "lmks5pts",
                "lmks68pts",
            ],
        )

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    results = []
    for result in tqdm(pool.map(func, natsorted(overlapnames)), total=len(overlapnames)):
        results.append(result)


@app.command()
def vfhq_directmhp_merge(
    directmhp_path: Path = typer.Argument(..., help="directmhp path", exists=True, dir_okay=True),
    gt_path: Path = typer.Argument(..., help="gt path", exists=True, dir_okay=True),
    out_path: Path = typer.Argument(..., help="out path"),
    iou_thresh: float = typer.Option(0.1, help="iou thresh"),
):
    gt_csvs = natsorted(gt_path.glob("*.csv"))
    out_path.mkdir(parents=True, exist_ok=True)
    for gt_csv in tqdm(gt_csvs):
        videoid = gt_csv.stem
        df_ = pd.read_csv(gt_csv)
        ref = {}
        # group dataframe by column name "frameid", all other columns should be aggregated
        df = df_.groupby("frameid", as_index=False).agg(
            {
                "idx": lambda x: list(x),
                "x1": lambda x: list(x),
                "y1": lambda x: list(x),
                "x2": lambda x: list(x),
                "y2": lambda x: list(x),
            }
        )
        records = df.to_dict("records")
        for row in records:
            frameid = row["frameid"]
            txtpath = directmhp_path / videoid / f"{str(frameid).zfill(8)}.txt"
            if not txtpath.exists():
                continue
            with open(txtpath) as f:
                lines = f.readlines()
            if len(lines) == 0:
                continue
            lines = [line.strip().split() for line in lines]
            yprltrbs = []
            for line in lines:
                yprltrbs.append(list(map(float, line)))
            pred_yprltrbs = np.array(yprltrbs)
            if len(line) == 1:
                ref[f"{frameid}_{ids[0]}"] = pred_yprltrbs[0, :3]
            else:
                ids = row["idx"]
                x1s = row["x1"]
                y1s = row["y1"]
                x2s = row["x2"]
                y2s = row["y2"]
                ltrbs = np.array(list(zip(x1s, y1s, x2s, y2s)))
                if (ltrbs[:, 2] < ltrbs[:, 0]).all():
                    ltrbs[:, 2] += ltrbs[:, 0]
                    ltrbs[:, 3] += ltrbs[:, 1]

                ious_ = ious(ltrbs, pred_yprltrbs[:, 3:])
                max_ids, max_ious = np.argmax(ious_, axis=-1), np.max(ious_, axis=-1)
                for i, (max_idx, max_iou) in enumerate(zip(max_ids, max_ious)):
                    if max_iou > iou_thresh:
                        ref[f"{frameid}_{ids[i]}"] = pred_yprltrbs[max_idx, :3]
        df_["mhp_yaw"] = np.nan
        df_["mhp_pitch"] = np.nan
        df_["mhp_roll"] = np.nan

        for idx, row in df_.iterrows():
            frameid = row["frameid"]
            fidx = row["idx"]
            key = f"{frameid}_{fidx}"
            if key in ref:
                df_.loc[idx, "mhp_yaw"] = ref[key][0]
                df_.loc[idx, "mhp_pitch"] = ref[key][1]
                df_.loc[idx, "mhp_roll"] = ref[key][2]
        outcsvpath = out_path / gt_csv.name
        df_.to_csv(
            outcsvpath.as_posix(),
            index=False,
            columns=[
                "frameid",
                "idx",
                "x1",
                "y1",
                "x2",
                "y2",
                "iqa",
                "synergy_yaw",
                "synergy_pitch",
                "synergy_roll",
                "poseanh_yaw",
                "poseanh_pitch",
                "poseanh_roll",
                "mhp_yaw",
                "mhp_pitch",
                "mhp_roll",
                "lmks5pts",
                "lmks68pts",
            ],
        )


@app.command()
def vfhq_combine_multiid_into_one(
    csv_basepath: Path = typer.Argument(..., help="csvpath", dir_okay=True, exists=True),
    out_basepath: Path = typer.Argument(..., help="outputpath"),
):
    csv_paths = list(csv_basepath.glob("*.csv"))
    csv_names = [x.stem for x in csv_paths]
    video_ids = list(x.split("+")[1] for x in csv_names)
    d = defaultdict(list)
    assert len(video_ids) == len(csv_paths)
    out_basepath.mkdir(parents=True, exist_ok=True)
    for i, (video_id, csv_path) in enumerate(zip(video_ids, csv_paths)):
        d[video_id].append(csv_path)

    for user_id, csv_paths in tqdm(d.items(), total=len(d)):
        dfs = []
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path.as_posix())
            df["user_id"] = user_id
            df["video_id"] = csv_path.stem
            dfs.append(df)
        result = pd.concat(dfs, ignore_index=True)
        outcsvpath = out_basepath / (user_id + ".csv")
        result.to_csv(outcsvpath.as_posix(), index=False)


@app.command()
def binning(
    csv_path: Path = typer.Argument(..., help="csvpath", exists=True, dir_okay=True),
    output_path: Path = typer.Argument(..., help="outpath"),
):
    csv_paths = list(natsorted(csv_path.glob("*.csv")))
    output_path.mkdir(exist_ok=True, parents=True)
    pbar = tqdm(csv_paths)
    hardcounter = defaultdict(int)
    softcounter = defaultdict(int)
    for csv_path in pbar:
        pbardesc = f"{csv_path.stem}"
        if "confused" in hardcounter:
            pbardesc += f"|H{hardcounter['confused']}"
        if "confused" in softcounter:
            pbardesc += f"|E{softcounter['confused']}"
        pbar.set_description(pbardesc)
        df = pd.read_csv(csv_path.as_posix())
        records = df.to_dict("records")
        for row_idx, row in enumerate(tqdm(records)):
            synergy_yaw = row["synergy_yaw"]
            synergy_pitch = row["synergy_pitch"]
            synergy_roll = row["synergy_roll"]
            poseanh_yaw = row["poseanh_yaw"]
            poseanh_pitch = row["poseanh_pitch"]
            poseanh_roll = row["poseanh_roll"]
            mhp_yaw = row["mhp_yaw"]
            mhp_pitch = row["mhp_pitch"]
            mhp_roll = row["mhp_roll"]

            synergy_bin = bin_a_pose(synergy_yaw, synergy_pitch, synergy_roll)
            poseanh_bin = bin_a_pose(poseanh_yaw, poseanh_pitch, poseanh_roll)
            mhp_bin = bin_a_pose(mhp_yaw, mhp_pitch, mhp_roll)
            df.loc[row_idx, "synergy_bin"] = synergy_bin
            df.loc[row_idx, "poseanh_bin"] = poseanh_bin
            df.loc[row_idx, "mhp_bin"] = mhp_bin

            candidates = []
            for c in [synergy_bin, poseanh_bin, mhp_bin]:
                if c is not None and not isinstance(c, float):
                    candidates.append(c)
            if len(candidates) == 0:
                hard_bin = "confused"
                soft_bin = "confused"
            else:
                is_allequal = all(element == candidates[0] for element in candidates)
                if is_allequal:
                    hard_bin = candidates[0]
                else:
                    hard_bin = "confused"

                is_profile = all([x.split("_")[0] == "profile" for x in candidates])
                is_frontal = all([x == "frontal" for x in candidates])
                counter = Counter(candidates)
                majority_vote = counter.most_common(1)[0]
                if majority_vote[1] > 1:
                    soft_bin = majority_vote[0]
                elif is_frontal:
                    soft_bin = "frontal"
                elif is_profile:
                    is_horizontal = all([x.split("_")[1] in ["left", "right", "extreme"] for x in candidates])
                    is_vertical = all([x.split("_")[1] in ["up", "down", "extreme"] for x in candidates])
                    if is_horizontal:
                        soft_bin = "profile_horizontal"
                    elif is_vertical:
                        soft_bin = "profile_vertical"
                    else:
                        soft_bin = "confused"
                else:
                    soft_bin = "confused"

            df.loc[row_idx, "hardbin"] = hard_bin
            df.loc[row_idx, "softbin"] = soft_bin
            hardcounter[hard_bin] += 1
            softcounter[soft_bin] += 1

        outputcsvpath = output_path / csv_path.name
        df.to_csv(outputcsvpath.as_posix(), index=False)

    print(f"Hard: {hardcounter}")
    print(f"Soft: {softcounter}")


@app.command()
def poseizer_stepbin(
    parquet_path: Path = typer.Argument(..., help="parquet path"),
    outparquet_path: Path = typer.Argument(..., help="outparquet path"),
):
    def func(row):
        posedict = get_pose_from_row(row)
        softbin = row["softbin"]
        vals = None
        if softbin in ["profile_left", "profile_right", "profile_horizontal"]:
            vals = list(posedict["yaw"].values())
        elif softbin in ["profile_up", "profile_down", "profile_vertical"]:
            vals = list(posedict["pitch"].values())
        elif softbin == "profile_extreme":
            vals_y = list(posedict["yaw"].values())
            vals_p = list(posedict["pitch"].values())
            vals_y_mean = [abs(x) for x in vals_y if not math.isnan(x)]
            vals_p_mean = [abs(x) for x in vals_p if not math.isnan(x)]
            vals_y_mean = sum(vals_y_mean) / len(vals_y_mean)
            vals_p_mean = sum(vals_p_mean) / len(vals_p_mean)
            if vals_y_mean > vals_p_mean:
                vals = vals_y
                softbin = "profile_horizontal"
            else:
                vals = vals_p
                softbin = "profile_vertical"
        if vals is not None:
            vals = list(filter(lambda x: not math.isnan(x), vals))
            final_val = 0
            for v in vals:
                final_val += abs(v)
            final_val /= len(vals)
            ranges = range(30, 91, 10)
            if final_val < 30:
                label = "frontal"
            else:
                for i, th in enumerate(ranges):
                    if final_val < th:
                        label = f"{ranges[i-1]}_{ranges[i]}"
                        break
                if final_val > th:
                    label = f"{ranges[-2]}_{ranges[-1]}"

            if "left" or "up" in softbin:
                final_val = -final_val
            if "horizontal" in softbin or "vertical" in softbin:
                n_positive = sum(x > 0 for x in vals)
                if n_positive > len(vals) // 2:
                    direction = "left" if softbin == "profile_horizontal" else "up"
                else:
                    direction = "right" if softbin == "profile_horizontal" else "down"
            else:
                direction = softbin.split("_")[-1]
            if label != "frontal":
                label = f"{direction}_{label}"
        else:
            label = softbin
        return label

    pandarallel.initialize(progress_bar=True)
    df = pd.read_parquet(parquet_path.as_posix())
    df["smallbin"] = df.parallel_apply(func, axis=1)
    outparquet_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_parquet(outparquet_path, index=False)


@app.command()
def csvs_to_parquet(
    csv_dir: Path = typer.Argument(..., help="input csvs"),
    output_path: Path = typer.Argument(..., help="output path"),
):
    csv_paths = csv_dir.rglob("*.csv")
    dfs = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        dfs.append(df)
    df = pd.concat(dfs)
    del df["idx"]
    output_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_parquet(output_path.as_posix())


@app.command()
def frontal_counter(
    parquet_path: Path = typer.Argument(..., help="parquet path"),
):
    df = pd.read_parquet(parquet_path)
    df = df[df["aligned_path"].notna()]
    df.sort_values(by="iqa", ascending=False, inplace=True)
    grouped = df.groupby(["video_id"])
    counter = 0
    for id_name, id_df in tqdm(grouped):
        poses = id_df["mhp_bin"].unique().tolist()
        if len(poses) == 1 and poses[0] == "frontal":
            print(id_name)
            counter += 1
    print(f"Total only frontal id: {counter}")


if __name__ == "__main__":
    app()
