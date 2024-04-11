import json
from pathlib import Path

import torch
import typer
from path_utils import get_filelist_and_cache
from PIL import Image
from tqdm.rich import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration


app = typer.Typer()


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


@app.command()
def main(
    imagedir: Path = typer.Argument(..., help="path to imagedir"),
    outpath: Path = typer.Argument(..., help="path to output", file_okay=True),
    bs: int = typer.Option(16, help="batchsize"),
    dry: bool = typer.Option(False, help="dryrun"),
):
    imagefiles = get_filelist_and_cache(imagedir, "*.[jp][pn]g")

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if dry:
        imagefiles = imagefiles[:10]
    chunks = list(divide_chunks(imagefiles, bs))
    res = {}
    for chunk in tqdm(chunks):
        prompt = ["this is a photograph of a"] * len(chunk)
        images = [Image.open(imagefile.as_posix()).convert("RGB") for imagefile in chunk]
        inputs = processor(images, text=prompt, return_tensors="pt").to(device, torch.float16)

        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        for imagefile, text in zip(chunk, generated_texts):
            res[imagefile.relative_to(imagedir).as_posix()] = text.strip()

    outpath.parent.mkdir(exist_ok=True, parents=True)
    with open(outpath.as_posix(), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    app()
