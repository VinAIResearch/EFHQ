import json
import shutil
from pathlib import Path

import typer
from tqdm import tqdm


app = typer.Typer(pretty_exceptions_show_locals=True)


@app.command()
def main(
    metadata_path: Path = typer.Argument(..., help="metadata path"),
    raw_basedir: Path = typer.Argument(..., help="Original Raw basedir"),
    align_basedir: Path = typer.Argument(..., help="Align basedir"),
    output_basedir: Path = typer.Argument(..., help="Output basedir"),
):
    with open(metadata_path) as f:
        lookup = json.load(f)
    detections_output_dir = output_basedir / "detections"
    detections_output_dir.mkdir(exist_ok=True, parents=True)
    for curfilename, d in tqdm(lookup.items(), total=len(lookup)):
        aligned_path = Path(d["original_path"])
        det_path = aligned_path.with_suffix(".txt")
        relative_path = aligned_path.relative_to(align_basedir)
        raw_parent_path = raw_basedir / relative_path.parent
        raw_img_path = raw_basedir / relative_path.name
        if not raw_img_path.exists():
            strips = relative_path.stem.split("_")
            assert len(strips) == 2
            filename = strips[0]
            raw_img_path = raw_parent_path / f"{filename}.png"
            assert raw_img_path.exists()
        dest_path = output_basedir / curfilename
        shutil.copy(raw_img_path, dest_path)
        shutil.copy(det_path, detections_output_dir / det_path.name)


if __name__ == "__main__":
    app()
