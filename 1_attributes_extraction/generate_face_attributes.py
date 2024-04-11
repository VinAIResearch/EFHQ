import json
from pathlib import Path

import typer
from deepface import DeepFace
from path_utils import get_filelist_and_cache
from tqdm import tqdm


app = typer.Typer()


def f(img_path, img_dir, outdir):
    json_path = outdir / img_path.relative_to(img_dir).with_suffix(".json")
    if json_path.exists():
        return
    objs = DeepFace.analyze(
        img_path=img_path.as_posix(),
        actions=["age", "gender", "race", "emotion"],
        enforce_detection=False,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path.as_posix(), "w") as f:
        json.dump(objs, f)


@app.command()
def main(
    img_dir: Path = typer.Argument(..., exists=True, dir_okay=True),
    outdir: Path = typer.Argument(..., dir_okay=True),
):
    img_paths = get_filelist_and_cache(img_dir, "*.[jp][pn]g")

    for img_path in tqdm(img_paths):
        f(img_path, img_dir, outdir)


if __name__ == "__main__":
    app()
