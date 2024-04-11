import json
import os
import shutil
from pathlib import Path

import pandas as pd
import typer
from tqdm import tqdm


app = typer.Typer()


@app.command()
def celeb(
    csv_dir: Path = typer.Argument(..., help="Base csv folder", dir_okay=True, exists=True),
    aligned_dir: Path = typer.Argument(..., help="Base aligned folder", dir_okay=True, exists=True),
    outdir: Path = typer.Argument(..., help="Output dir"),
    image_per_idbin: int = typer.Option(10, help="How much sample per id_bin"),
    iqa_threshold: float = typer.Option(42, help="IQA threshold"),
):
    pose_dirs = set(os.listdir(csv_dir))
    if "frontal" in pose_dirs:
        pose_dirs.remove("frontal")
    outdir.mkdir(exist_ok=True, parents=True)
    outimgdir = outdir / "images"
    outmetafile = outdir / "metadata.json"
    outimgdir.mkdir(exist_ok=True, parents=True)
    pose_dirs = sorted(pose_dirs)
    info = {}
    image_counter = 0
    for pose_dir in tqdm(pose_dirs, desc="Pose folder"):
        pose_dir = csv_dir / pose_dir
        csv_files = list(pose_dir.rglob("*.csv"))
        for csv_file in tqdm(csv_files, desc="Parse csv"):
            df = pd.read_csv(csv_file)
            df = df[df["iqa"] >= iqa_threshold]
            if len(df) > image_per_idbin:
                df["abs_synergy_yaw"] = df["synergy_yaw"].abs()
                df["abs_synergy_pitch"] = df["synergy_pitch"].abs()
                df.sort_values(by=["abs_synergy_yaw", "abs_synergy_pitch", "iqa"], inplace=True)
                df = df.head(image_per_idbin)
            for row in df.itertuples():
                frame_idx = row.frameid
                image_name = f"{str(image_counter).zfill(8)}.png"
                original_path = aligned_dir / csv_file.stem / f"{str(frame_idx).zfill(8)}_0.png"
                if not original_path.exists():
                    print(f"{original_path} not exist")
                else:
                    shutil.copy2(original_path, outimgdir / image_name)
                    image_counter += 1
                    info[image_name] = {
                        "original_path": original_path.as_posix(),
                        "original_csv": csv_file.as_posix(),
                        "row_index": int(row.Index),
                    }
    with open(outmetafile, "w") as f:
        json.dump(info, f)


@app.command()
def vfhq(
    parquet_path: Path = typer.Argument(..., help="parquet path", file_okay=True, exists=True),
    aligned_dir: Path = typer.Argument(..., help="Base aligned folder", dir_okay=True, exists=True),
    outdir: Path = typer.Argument(..., help="Output dir"),
    image_per_idbin: int = typer.Option(10, help="How much sample per id_bin"),
    iqa_threshold: float = typer.Option(42, help="IQA threshold"),
):
    def sort_and_sample(group, x):
        abs_yaw = group["synergy_yaw"].abs()
        abs_pitch = group["synergy_pitch"].abs()

        group["abs_synergy_yaw"] = abs_yaw
        group["abs_synergy_pitch"] = abs_pitch

        sorted_group = group.sort_values(
            by=["abs_synergy_yaw", "abs_synergy_pitch", "iqa"], ascending=[False, False, False]
        )

        filtered_group = sorted_group[sorted_group["iqa"] >= iqa_threshold]
        return filtered_group.head(x)

    df = pd.read_parquet(parquet_path)
    outdir.mkdir(exist_ok=True, parents=True)
    outimgdir = outdir / "images"
    outmetafile = outdir / "metadata.json"
    outimgdir.mkdir(exist_ok=True, parents=True)
    sampled_data = df.groupby(["video_id", "softbin"], group_keys=False).apply(
        lambda x: sort_and_sample(x, image_per_idbin)
    )

    info = {}
    image_counter = 0
    for row in tqdm(sampled_data.itertuples(), total=len(sampled_data), desc="Copying"):
        frame_idx = row.frameid
        image_name = f"{str(image_counter).zfill(8)}.png"
        original_path = aligned_dir / row.video_id / f"{str(frame_idx).zfill(8)}_0.png"
        if not original_path.exists():
            print(f"{original_path} not exist")
        else:
            shutil.copy2(original_path, outimgdir / image_name)
            image_counter += 1
            info[image_name] = {
                "original_path": original_path.as_posix(),
                "original_csv": parquet_path.as_posix(),
                "row_index": int(row.Index),
            }
    with open(outmetafile, "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    app()
