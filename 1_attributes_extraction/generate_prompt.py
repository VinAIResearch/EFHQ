import json
from pathlib import Path
from typing import Dict, List

import typer
from pydantic import BaseModel


class Region(BaseModel):
    x: int
    y: int
    w: int
    h: int


class FaceAttributes(BaseModel):
    age: int
    region: Region
    dominant_gender: str
    race: Dict
    dominant_race: str
    emotion: Dict
    dominant_emotion: str


def select_biggest(attribs: List[FaceAttributes]):
    biggest_area = -1
    biggest_idx = 0
    for i, attrib in enumerate(attribs):
        area = attrib.region.w * attrib.region.h
        if area > biggest_area:
            biggest_area = area
            biggest_idx = i
    return attribs[biggest_idx]


app = typer.Typer()


@app.command()
def main(
    attributes_dir: Path = typer.Argument(..., exists=True, dir_okay=True),
    outpath: Path = typer.Argument(..., file_okay=True),
):
    json_files = list(attributes_dir.rglob("*.json"))
    res = {}
    for json_file in json_files:
        with open(json_file) as f:
            attribs = json.load(f)

        name = json_file.stem
        if "mirror" in name:
            continue
        folder = json_file.parent.name

        attribs = [FaceAttributes.parse_obj(x) for x in attribs]
        if len(attribs) > 1:
            attrib = select_biggest(attribs)
            subject = attrib.dominant_gender.lower()
            emotion = attrib.dominant_emotion.lower() + " "
            race = attrib.dominant_race.lower() + " "
        elif len(attribs) == 0:
            subject = "person"
            emotion = ""
            race = ""
        else:
            attrib = attribs[0]
            subject = attrib.dominant_gender.lower()
            emotion = attrib.dominant_emotion.lower() + " "
            race = attrib.dominant_race.lower() + " "
        # add case norm
        prompt = f"A profile portrait image of a {emotion}{race}{subject}."
        res[f"{folder}/{name}"] = prompt
        # add case mirror
        name += "_mirror"
        prompt = f"A portrait image of a {emotion}{race}{subject}."
        res[f"{folder}/{name}"] = prompt

    outpath.parent.mkdir(exist_ok=True, parents=True)
    with open(outpath, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    app()
