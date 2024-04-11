import json
import shutil
from pathlib import Path

import typer
from natsort import natsorted
from tqdm import tqdm


app = typer.Typer()


@app.command()
def merge_eg3d(
    image_dir: Path = typer.Argument(..., help="FFHQ dir", dir_okay=True, exists=True),
    ffhq_dataset_json_path: Path = typer.Argument(..., help="FFHQ's dataset.json path", file_okay=True, exists=True),
    celeb_image_path: Path = typer.Argument(..., help="Aligned CelebVHQ path", file_okay=True, exists=True),
    celeb_json_path: Path = typer.Argument(..., help="CelebVHQ's dataset.json path", file_okay=True, exists=True),
    celeb_filter_path: Path = typer.Option(None, help="CelebVHQ's filterlist path", file_okay=True, exists=True),
    vfhq_image_path: Path = typer.Argument(..., help="Aligned VFHQ path", file_okay=True, exists=True),
    vfhq_json_path: Path = typer.Argument(..., help="VFHQ's dataset.json path", file_okay=True, exists=True),
    vfhq_filter_path: Path = typer.Option(None, help="VFHQ's filterlist path", file_okay=True, exists=True),
    start_fileidx: int = typer.Option(139914, help="Start Image file Index (from FFHQ)"),
    start_folderidx: int = typer.Option(139, help="Start folder index (from FFHQ)"),
):
    def read_labelobj_to_dict(path):
        with open(path) as f:
            label_obj = json.load(f)
        d = {}
        for label in label_obj["labels"]:
            d[label[0]] = label[1]
        return d

    with open(ffhq_dataset_json_path) as f:
        label_objs = json.load(f)
    celeb_lookup = read_labelobj_to_dict(celeb_json_path)
    vfhq_lookup = read_labelobj_to_dict(vfhq_json_path)

    celeb_filter = set()
    vfhq_filter = set()
    if celeb_filter_path is not None:
        with open(celeb_filter_path) as f:
            celeb_filter = set(map(lambda x: x.strip(), f.readlines()))
    if vfhq_filter_path is not None:
        with open(vfhq_filter_path) as f:
            vfhq_filter = set(map(lambda x: x.strip(), f.readlines()))

    print(f"celeb filter len: {len(celeb_filter)}")
    print(f"vfhq filter len: {len(vfhq_filter)}")
    celeb_image_paths = natsorted(list(celeb_image_path.glob("*.png")))
    cur_file_idx = start_fileidx
    lookup = {}
    for file in tqdm(celeb_image_paths, desc="copy celeb"):
        if file.name in celeb_filter:
            continue
        if cur_file_idx % 1000 == 0:
            start_folderidx += 1
        parent_name = str(start_folderidx).zfill(5)
        fname = str(cur_file_idx).zfill(8)
        fname = f"{parent_name}/img{fname}.png"
        dest = image_dir / fname
        dest.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(file, dest)
        label_objs["labels"].append([fname, celeb_lookup[file.name]])
        cur_file_idx += 1
        lookup["celeb/" + file.name] = fname

    vfhq_image_paths = natsorted(list(vfhq_image_path.glob("*.png")))
    for file in tqdm(vfhq_image_paths, desc="copy vfhq"):
        if file.name in vfhq_filter:
            continue
        if cur_file_idx % 1000 == 0:
            start_folderidx += 1
        parent_name = str(start_folderidx).zfill(5)
        fname = str(cur_file_idx).zfill(8)
        fname = f"{parent_name}/img{fname}.png"
        dest = image_dir / fname
        dest.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(file, dest)
        label_objs["labels"].append([fname, vfhq_lookup[file.name]])
        cur_file_idx += 1
        lookup["vfhq/" + file.name] = fname

    with open(image_dir / "dataset_merged.json", "w") as f:
        json.dump(label_objs, f, indent=2)
    with open(image_dir / "lookup.json", "w") as f:
        json.dump(lookup, f, indent=2)


if __name__ == "__main__":
    app()
