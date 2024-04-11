import hashlib
import pickle
from pathlib import Path
from typing import List

from natsort import natsorted


CACHE_BASEDIR = Path("~/.cache/efhq").expanduser()
SECRET = "efhq!@#!"
CACHE_PATH_DIR = CACHE_BASEDIR / "filelists"
CACHE_PATH_DIR.mkdir(exist_ok=True, parents=True)


def generate_cachepath_from_path(path: Path) -> Path:
    path = path.resolve().as_posix()
    md5sum = hashlib.md5(path.encode("utf-8")).hexdigest()
    return CACHE_PATH_DIR / md5sum


def get_filelist_and_cache(path: Path, glob_str: str) -> List[Path]:
    pathstr = path.resolve().as_posix()
    pathstr_ = pathstr + f"{SECRET}_glob_{glob_str}"
    md5sum = hashlib.md5(pathstr_.encode("utf-8")).hexdigest()
    cachepath = CACHE_PATH_DIR / f"{md5sum}.pkl"
    reload = True
    filelist = []
    if cachepath.exists():
        k = input(f"Found filelist cache for {pathstr}, load it? (y/n)")
        if k.lower() and k == "y":
            with open(cachepath, "rb") as f:
                filelist = pickle.load(f)
                reload = False
        else:
            reload = True
    if reload:
        filelist = natsorted(list(path.rglob(glob_str)))
        with open(cachepath, "wb") as f:
            pickle.dump(filelist, f)
    return filelist
