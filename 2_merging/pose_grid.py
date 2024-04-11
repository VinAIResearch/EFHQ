import base64
import math
import os
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
import typer
from natsort import natsorted
from PIL import Image
from st_clickable_images import clickable_images


CACHE_PATH = Path("./tmp/posegrid")
app = typer.Typer()


@st.cache()
def globs(basepath: Path, pattern: str):
    files = natsorted(list(basepath.rglob(pattern)))
    return files


def bin_to_color(bin):
    d = {
        "frontal": "white",
        "profile_left": "red",
        "profile_right": "green",
        "profile_up": "yellow",
        "profile_down": "blue",
        "profile_extreme": "purple",
        "confused": "orange",
        "profile_horizontal": "cyan",
        "profile_vertical": "grey",
        "remove": "pink",
    }
    return d[bin]


@st.cache()
def read_img_from_zip(zip_path: Path, fids):
    def get_base64_of_image(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("ascii")

    def get_thumbnail_base64(file_path, size=(50, 50)):
        with Image.open(file_path) as image:
            image.thumbnail(size)
            with BytesIO() as buffer:
                image.save(buffer, "JPEG")
                return base64.b64encode(buffer.getvalue()).decode()

    res = {}
    with zipfile.ZipFile(str(zip_path), "r") as zip_file:
        if len(zip_file.namelist()) == 1:
            return res
        for ori_file_name in zip_file.namelist()[1:]:
            file_name = ori_file_name.split("/")[-1]
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                continue
            fn = Path(file_name)
            if fn.stem not in fids:
                print("Not exist: ", fn.stem)
                continue
            with zip_file.open(ori_file_name) as my_file:
                image_bytes = my_file.read()
                with Image.open(BytesIO(image_bytes)) as img:
                    img.thumbnail((250, 250))
                    img_resized_bytes = BytesIO()
                    img.save(img_resized_bytes, format="JPEG")
                    img_base64 = base64.b64encode(img_resized_bytes.getvalue()).decode()
                res[Path(file_name).stem] = img_base64
    return res


@st.cache()
def read_img_from_folder(folder_path: Path, fids):
    res = {}
    images_path = folder_path.glob("*.png")
    for image_path in images_path:
        if image_path.stem not in fids:
            continue
        with Image.open(image_path.as_posix()) as img:
            img.thumbnail((100, 100))
            img_resized_bytes = BytesIO()
            img.save(img_resized_bytes, format="JPEG")
            img_base64 = base64.b64encode(img_resized_bytes.getvalue()).decode("utf-8")
        res[image_path.stem] = img_base64
    return res


if "csv_counter" not in st.session_state:
    st.session_state.csv_counter = 0


def save(save_inplace, csvs):
    if save_inplace and "df" in st.session_state:
        if "changes" in st.session_state and st.session_state.changes:
            current_csv = csvs[st.session_state.csv_counter]
            st.session_state.df.to_csv(current_csv, index=False)
            st.session_state.changes = False


def process_csv(save_inplace, csvs, category_path):
    if save_inplace:
        current_csv = csvs[st.session_state.csv_counter]
        out_dir = str(category_path)

        id_name = os.path.basename(current_csv).split(".csv")[0]
        df = st.session_state.df.copy(deep=True)

        remove = ["confused", "trash", "remove", "profile_vertical", "profile_horizontal"]
        df = df.drop(df[df["softbin"].isin(remove)].index)

        category = df.groupby("softbin")
        for label, group in category:
            save_dir = os.path.join(out_dir, label)
            os.makedirs(save_dir, exist_ok=True)
            group.to_csv(f"{save_dir}/{id_name}.csv", index=False)


@app.command()
def main(
    csv_paths: Path = typer.Argument(..., help="csv path"),  # Folder containing original csv
    zip_paths: Path = typer.Argument(..., help="zip path"),  # Folder containing zip
    posebin: str = typer.Option("softbin", help="column name for posebin"),
    save_csv_tmp: Path = typer.Argument(..., help="csv path"),  # Folder to save corrected csv
    category_path: Path = typer.Argument(..., help="final csv path"),
):
    st.title("Pose Grid")
    all_csv_name = os.listdir(csv_paths.as_posix())
    all_csv_name = sorted(i.strip() for i in all_csv_name)
    all_csv_name = [i for i in all_csv_name if ".csv" in i]
    csvs = [csv_paths / i for i in all_csv_name]

    os.makedirs(save_csv_tmp.as_posix(), exist_ok=True)
    os.makedirs(category_path.as_posix(), exist_ok=True)
    local_csvs = [save_csv_tmp / i for i in all_csv_name]

    if "csv_counter" not in st.session_state:
        st.session_state.csv_counter = 0

    with st.sidebar:
        save_inplace = st.checkbox("Save inplace?")

        filter_box = st.sidebar.selectbox(
            "Filter",
            (
                "all",
                "profile_horizontal",
                "profile_vertical",
                "profile_left",
                "profile_right",
                "profile_up",
                "profile_down",
                "frontal",
                "profile_extreme",
                "confused",
            ),
        )
        new_label = st.sidebar.selectbox(
            "New Label",
            (
                "frontal",
                "profile_up",
                "profile_down",
                "profile_left",
                "profile_right",
                "profile_extreme",
                "remove",
            ),
        )

        index_check = st.text_input("Index", st.session_state.csv_counter)
        index_check = int(index_check)
        assert index_check < len(csvs)
        st.session_state.csv_counter = index_check

        current_csv = csvs[st.session_state.csv_counter]
        local_csv = local_csvs[st.session_state.csv_counter]
        if not os.path.exists(str(local_csv)):
            print("Download:", {str(current_csv)})
            os.system(f"cp {current_csv} {save_csv_tmp}")

        st.title(f"ID: {local_csv.stem} ({st.session_state.csv_counter + 1}/{len(csvs)})")
        img_size = st.slider("Image Size", 100, 200, 50, step=5, key="img_size")
        all_change_idx_box = prompt()

        if st.button("Set all to label"):
            st.session_state.changes = True
            if filter_box != "all":
                st.session_state.df.loc[st.session_state.df[posebin] == filter_box, posebin] = new_label
            else:
                st.session_state.df.loc[:, posebin] = new_label
            save(save_inplace, local_csvs)
            st.experimental_rerun()

        if st.button("Next"):
            save(save_inplace, local_csvs)
            process_csv(save_inplace, local_csvs, category_path)

            st.session_state.changes = False
            st.session_state.csv_counter = min(st.session_state.csv_counter + 1, len(csvs) - 1)
            st.experimental_rerun()

        if st.button("Prev"):
            save(save_inplace, local_csvs)
            st.session_state.changes = False
            st.session_state.csv_counter = max(st.session_state.csv_counter - 1, 0)
            st.experimental_rerun()

        if st.button("Find first have image"):
            save(save_inplace, local_csvs)
            st.session_state.csv_counter += 1
            st.session_state.changes = False
            while st.session_state.csv_counter < len(csvs) - 1:
                current_csv = csvs[st.session_state.csv_counter]
                local_csv = local_csvs[st.session_state.csv_counter]
                if not os.path.exists(str(local_csv)):
                    os.system(f"cp {current_csv} {save_csv_tmp}")
                df = pd.read_csv(local_csv)
                st.session_state.df = df
                filtered_df = df[df[posebin] == filter_box] if filter_box != "all" else df
                if len(filtered_df) == 0:
                    st.session_state.csv_counter = min(st.session_state.csv_counter + 1, len(csvs) - 1)
                else:
                    st.experimental_rerun()
            else:
                st.text(f"All ids don't have {filter_box} bin")

    current_csv = csvs[st.session_state.csv_counter]
    local_csv = local_csvs[st.session_state.csv_counter]
    if not os.path.exists(str(local_csv)):
        print(f"cp {current_csv} {save_csv_tmp}")
        os.system(f"cp {current_csv} {save_csv_tmp}")

    df = pd.read_csv(local_csv)
    st.session_state.df = df

    df = st.session_state.df
    current_zip = zip_paths / f"{local_csv.stem}.zip"

    filtered_df = df[df[posebin] == filter_box] if filter_box != "all" else df.copy()
    filtered_df.reset_index(inplace=True)

    fids = filtered_df["frameid"].tolist()
    fids = set(map(lambda x: str(x).zfill(8), fids))

    if len(filtered_df) == 0:
        st.text("No images")
    else:
        images_dict = read_img_from_zip(current_zip, fids)
        if len(images_dict.keys()) == 0:
            st.text("No images")
        else:
            dict_df = filtered_df.to_dict("records")
            images = []
            colors = []
            for d in dict_df:
                fid = str(d["frameid"]).zfill(8)
                if fid not in images_dict.keys():
                    continue
                images.append(images_dict[fid])
                if filter_box == "profile_horizontal":
                    if d["hardbin"] in ["profile_left", "profile_right"]:
                        bin = d["hardbin"]
                    l = [-d["synergy_yaw"], d["poseanh_yaw"]]
                    if isinstance(d["mhp_yaw"], float) and not math.isnan(d["mhp_yaw"]):
                        l.append(d["mhp_yaw"])
                    is_right = sum(x > 0 for x in l)
                    is_left = sum(x < 0 for x in l)
                    if is_right > is_left:
                        bin = "profile_right"
                    else:
                        bin = "profile_left"

                elif filter_box == "profile_vertical":
                    if d["hardbin"] in ["profile_up", "profile_down"]:
                        bin = d["hardbin"]

                    l = [-d["synergy_pitch"], d["poseanh_pitch"]]
                    if isinstance(d["mhp_pitch"], float) and not math.isnan(d["mhp_pitch"]):
                        l.append(d["mhp_pitch"])
                    is_up = sum(x > 0 for x in l)
                    is_down = sum(x < 0 for x in l)
                    if is_up > is_down:
                        bin = "profile_up"
                    else:
                        bin = "profile_down"
                else:
                    bin = d[posebin]
                color = bin_to_color(bin)
                colors.append(color)
                color_indexes = {}
                for i, color in enumerate(colors):
                    if color not in color_indexes:
                        color_indexes[color] = [i]
                    else:
                        color_indexes[color].append(i)

            clicks = show_grid_of_images(images, color_indexes, img_size)

            for k in list(clicks.keys()):
                if clicks[k]["click"] > -1:
                    st.session_state.changes = True
                    location = clicks[k]["mapping"][clicks[k]["click"]]
                    clicked_index = filtered_df.iloc[location]["index"]
                    st.session_state.df.loc[clicked_index, posebin] = new_label
                    save(save_inplace, local_csvs)
                    st.experimental_rerun()

            if st.button("Save"):
                for k in list(clicks.keys()):
                    if all_change_idx_box[k] != "NA" and all_change_idx_box[k] != "":
                        st.session_state.changes = True
                        change_idx = [int(i) for i in all_change_idx_box[k].strip().split(",")]
                        assert len(change_idx) == 2
                        change_idx = [int(i) for i in range(change_idx[0], change_idx[1] + 1, 1)]

                        for i in change_idx:
                            location = clicks[k]["mapping"][i]
                            clicked_index = filtered_df.iloc[location]["index"]

                            st.session_state.df.loc[clicked_index, posebin] = new_label
                    all_change_idx_box[k] = "NA"
                save(save_inplace, local_csvs)
                all_change_idx_box
                st.experimental_rerun()

            if st.button("Next"):
                save(save_inplace, local_csvs)
                process_csv(save_inplace, local_csvs, category_path)

                st.session_state.changes = False
                st.session_state.csv_counter = min(st.session_state.csv_counter + 1, len(csvs) - 1)
                st.experimental_rerun()

            if st.button("Prev"):
                save(save_inplace, local_csvs)
                st.session_state.changes = False
                st.session_state.csv_counter = max(st.session_state.csv_counter - 1, 0)
                st.experimental_rerun()


# Load the images and display them in a grid
def show_grid_of_images(image_files, color_indexes, img_size):
    clicks = {}
    for color, index in color_indexes.items():
        new_images = [image_files[i] for i in index]
        mapping = {new_index: old_index for new_index, old_index in enumerate(index)}
        images = []
        for file in new_images:
            images.append(f"data:image/jpeg;base64,{file}")

        clicks[f"{color}"] = {
            "click": clickable_images(
                images,
                titles=[i for i in range(len(images))],
                div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
                img_style={
                    "margin": "3px",
                    "height": f"{img_size}px",
                    "width": f"{img_size}px",
                    "border": f"3px solid {color}",
                },
            ),
            "mapping": mapping,
        }

    return clicks


def show_grid_of_images2(image_files, colors, img_size):
    images = []
    for file in image_files:
        images.append(f"data:image/jpeg;base64,{file}")

    clicked = clickable_images(
        images,
        titles=[i for i in range(len(image_files))],
        div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
        img_style={
            "margin": "1px",
            "height": f"{img_size}px",
            "width": f"{img_size}px",
            "border": f"3px solid {colors[0]}",
        },
    )
    return clicked


def prompt():
    d = {
        "frontal": "white",
        "confused": "orange",
        "profile_left": "red",
        "profile_right": "green",
        "profile_up": "yellow",
        "profile_down": "blue",
        "profile_extreme": "purple",
        "profile_horizontal": "cyan",
        "profile_vertical": "grey",
        "remove": "pink",
    }
    all_change_idx_box = {}
    for k, v in d.items():
        all_change_idx_box[v] = st.text_input(f"Index {v}", "NA")

    return all_change_idx_box


if __name__ == "__main__":
    app(standalone_mode=False)
