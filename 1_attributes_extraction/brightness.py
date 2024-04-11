import math
from pathlib import Path

import pandas as pd
import typer
from PIL import Image, ImageStat
from tqdm import tqdm


def brightness(im_file: Path):
    preset1 = (0.299, 0.587, 0.114)
    preset = preset1

    im = Image.open(im_file.as_posix())
    stat = ImageStat.Stat(im)
    r, g, b = stat.mean
    return math.sqrt(preset[0] * (r**2) + preset[1] * (g**2) + preset[2] * (b**2))


app = typer.Typer()


@app.command()
def main(
    input_dir: Path = typer.Argument(..., help="path to input dir", file_okay=True, dir_okay=True, exists=True),
    output_dir: Path = typer.Argument(..., help="path to out dir"),
    dry_run: bool = typer.Option(False, help="run test"),
):
    image_paths = list(input_dir.rglob("*.[pj][pn]g"))
    d = {}
    for i, image_path in enumerate(tqdm(image_paths)):
        if dry_run and i > 100:
            break
        brightness_val = brightness(image_path)
        d[image_path.relative_to(input_dir).as_posix()] = brightness_val
    d_ = {"path": list(d.keys()), "brightness": list(d.values())}
    df = pd.DataFrame.from_dict(d_)
    df.to_csv(output_dir.as_posix(), index=False)


if __name__ == "__main__":
    app()
