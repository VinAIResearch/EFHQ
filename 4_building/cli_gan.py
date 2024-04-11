import json
from pathlib import Path

import pandas as pd
import typer
from tqdm.rich import tqdm


app = typer.Typer()


@app.command()
def make_dataset_json(
    compose_basepath: Path = typer.Argument(..., help="Compose dir", dir_okay=True, exists=True),
    ffhqpose_path: Path = typer.Argument(..., help="FFHQ Pose path from FFHQAging", file_okay=True, exists=True),
):
    df = pd.read_csv(ffhqpose_path)

    img_paths = list(compose_basepath.rglob("*.png"))
    res = {"labels": []}
    for img_path in tqdm(img_paths):
        if "ffhq" in img_path.parent.name.lower():
            img_number = int(img_path.stem)
            row = df[df["image_number"] == img_number].iloc[0]
            p, _, y = row["head_pitch"], row["head_roll"], row["head_yaw"]
            if abs(p) > 30 or abs(y) > 45:
                label = "extreme"
            else:
                label = "frontal"
        else:
            label = "extreme"
        labelstr_to_idx = {"frontal": 0, "extreme": 1}
        label = int(labelstr_to_idx[label])
        res["labels"].append((img_path.relative_to(compose_basepath).as_posix(), label))
    outpath = compose_basepath / "dataset.json"
    if outpath.exists():
        p = input(f"{outpath} exists, override?(y/n)")
        if p.lower() != "y":
            exit()
    with open(outpath, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    app()
