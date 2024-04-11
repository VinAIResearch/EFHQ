- [Install](#install)
- [Download dataset](#download-dataset)
  - [CelebVHQ](#celebvhq)
  - [VFHQ](#vfhq)
- [Attributes extraction](#attributes-extraction)
  - [CelebV-HQ](#celebv-hq)
    - [Face bounding boxes + 5 keypoint landmarks](#face-bounding-boxes--5-keypoint-landmarks)
    - [Pose](#pose)
      - [SynergyNet](#synergynet)
      - [FacePoseNet](#faceposenet)
      - [DirectMHP](#directmhp)
    - [HyperIQA](#hyperiqa)
  - [VFHQ](#vfhq-1)
    - [Face bounding boxes + 5 keypoint landmarks](#face-bounding-boxes--5-keypoint-landmarks-1)
    - [Pose](#pose-1)
      - [SynergyNet](#synergynet-1)
      - [FacePoseNet](#faceposenet-1)
      - [DirectMHP](#directmhp-1)
    - [HyperIQA](#hyperiqa-1)
  - [Dataset neutral attributes](#dataset-neutral-attributes)
    - [Brightness](#brightness)
- [Attributes unification](#attributes-unification)
  - [CelebV-HQ](#celebv-hq-1)
  - [VFHQ](#vfhq-2)
- [Aligning](#aligning)
  - [StyleGAN](#stylegan)
  - [EG3D](#eg3d)
  - [Face Verification](#face-verification)
- [Sampling](#sampling)
  - [Face Generation](#face-generation)
    - [StyleGAN](#stylegan-1)
    - [EG3D](#eg3d-1)
  - [Face Verification](#face-verification-1)
- [Building final dataset](#building-final-dataset)
  - [StyleGAN](#stylegan-2)
  - [EG3D](#eg3d-2)
  - [ControlNet](#controlnet)
  - [Face Reenactment](#face-reenactment)
    - [Vox Preprocessing](#vox-preprocessing)
    - [LightFace for Celeb](#lightface-for-celeb)
    - [Reorganize CelebV-Reenactment data](#reorganize-celebv-reenactment-data)
    - [Get text data file for reenactment data](#get-text-data-file-for-reenactment-data)
- [Training](#training)
  - [Face synthesis](#face-synthesis)
    - [StyleGAN](#stylegan-3)
    - [EG3D](#eg3d-3)
    - [ControlNet](#controlnet-1)
  - [Face Reenactment](#face-reenactment-1)
    - [TPS](#tps)
    - [LIA](#lia)
- [Evaluation](#evaluation)
  - [Face Synthesize](#face-synthesize)
    - [StyleGAN](#stylegan-4)
    - [EG3D](#eg3d-4)
    - [ControlNet](#controlnet-2)
  - [Face Reenactment](#face-reenactment-2)
    - [TPS](#tps-1)
    - [LIA](#lia-1)
  - [Face Verification](#face-verification-2)

# Install

- Make sure you have `ffmpeg` in your `$PATH`

```bash
conda create -p ./env python=3.8
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

# Download dataset

## CelebVHQ

Given the id list path is `<id_all.txt>` and the annotation of CelebVFHQ located at `repodir/data/CelebV-HQ/annotations/celebvhq_info.json`, one can chunk the filelist into multiple txt file and run the following:

```bash
cd 0_dataset_download

python download_celeb.py \
	--annotation <id_all.txt> \
	--metadata ../data/CelebV-HQ/annotations/celebvhq_info.json \
	--save_folder ../data/CelebV-HQ \
	--type all \
	--check_point ../data/CelebV-HQ/ckpt/frame/extract_ckpt_all.txt \
	--process 0

python download_celeb.py \
	--annotation <id_all.txt> \
	--metadata ../data/CelebV-HQ/annotations/celebvhq_info.json \
	--save_folder ../data/CelebV-HQ \
	--type all \
	--check_point ../data/CelebV-HQ/ckpt/frame/extract_ckpt_all.txt \
	--process 1
```

- If possible, zip each video folder into zip file to further optimize I/O utilization

## VFHQ

Given the id list is `<id_train.txt>`, `<id_test.txt>` and the original annotation folder at `repodir/data/VFHQ/annotations`, one can chunk the filelist into multiple txt file and run the following:

```bash
cd 0_dataset_download
python download_vfhq.py \
	--annotation <id_train.txt> \
	--annotation_folder ../data/VFHQ/annotations \
	--save_folder ../data/VFHQ \
	--type train \
	--check_point ../data/VFHQ/ckpt/frame/ckpt_train.txt

python download_vfhq.py \
	--annotation <id_test.txt> \
	--annotation_folder ../data/VFHQ/annotations \
	--save_folder ../data/VFHQ \
	--type test \
	--check_point ../data/VFHQ/ckpt/frame/ckpt_test.txt
```

- We also zip each video folder into zip file to further optimize I/O utilization

# Attributes extraction

## CelebV-HQ

### Face bounding boxes + 5 keypoint landmarks

```bash
cd 1_attributes_extraction
python detect_face.py \
	--source_folder ../data/CelebV-HQ/processed/extracted_hq_results/all \
	--save_folder ../data/CelebV-HQ/info/retina
```

### Pose

#### SynergyNet

```bash
mkdir -p vendors
git clone https://github.com/choyingw/SynergyNet vendors/synergynet
# please setup to make sure you can run synergynet
cp 1_attributes_extraction/synergy_celeb.py vendors/synergynet/synergy_celeb.py
cd vendors/synergynet/
python synergy_celeb.py \
    --id_text <id_all.txt> \
    --reference_folder ../../data/CelebV-HQ/info/retina \
    --source_folder ../../data/CelebV-HQ/processed/extracted_hq_results \
    --type all \
    --save_folder ../../data/CelebV-HQ/info/synergy \
    --name procname \
    --num_process 5
```

#### FacePoseNet

- Since this is our inhouse models, we will only provide the pose estimation output.

#### DirectMHP

```bash
git clone https://github.com/hnuzhy/DirectMHP vendors/DirectMHP
# please setup to make sure you can run DirectMHP
cp 1_attributes_extraction/directmhp_infercli.py vendors/DirectMHP/infercli.py
cp 1_attributes_extraction/directmhp_zipdatasetcustom.py vendors/DirectMHP/zipdatasetcustom.py
cd vendors/DirectMHP/
# for each subsetlist `ziplist_x.txt` (you can also input the basedir, it will automatically glob all zip files), run the following
python infercli.py \
	<ziplist_txt_path> \
	../../data/CelebV-HQ/info/directmhp
```

### HyperIQA

```bash
git clone https://github.com/SSL92/hyperIQA vendors/hyperIQA
# please setup to make sure you can run hyperIQA
cp 1_attributes_extraction/iqa_celeb.py vendors/hyperIQA/
cd vendors/hyperIQA/
python iqa_celeb.py \
    --name all \
    --id_text <id_all.txt> \
    --annotation_folder ../../data/CelebV-HQ/info/retina \
    --source_folder ../../data/CelebV-HQ/processed/extracted_hq_results \
    --type all \
    --save_folder ../../data/CelebV-HQ/info/hyperiqa \
    --num_process 2
```

## VFHQ

### Face bounding boxes + 5 keypoint landmarks

- These attributes are provided by the original dataset

### Pose

#### SynergyNet

```bash
cp 1_attributes_extraction/synergy_vfhq.py vendors/synergynet/synergy_vfhq.py
cd vendors/synergynet/
python synergy_vfhq.py \
		--name test \
		--id_text <id_test.txt> \
		--annotation_folder ../../data/VFHQ/annotations \
		--source_folder ../../data/VFHQ/processed/extracted_hq_results \
		--type test \
		--save_folder ../../data/VFHQ/info/synergy \
		--num_process 2

python synergy_vfhq.py \
		--name train \
		--id_text <id_train.txt> \
		--annotation_folder ../../data/VFHQ/annotations \
		--source_folder ../../data/VFHQ/processed/extracted_hq_results \
		--type train \
		--save_folder ../../data/VFHQ/info/synergy \
		--num_process 2
```

#### FacePoseNet

- Since this is our inhouse models, we will only provide the pose estimation output.

#### DirectMHP

```bash
cd vendors/DirectMHP/
# for each `<ziplist_txt_path>` (you can also input the basedir, it will automatically glob all zip files), run the following
python infercli.py \
	<{train|test}ziplist_txt_path> \
	../../data/VFHQ/info/directmhp/{train|test}
```

### HyperIQA

```bash

cp 1_attributes_extraction/iqa_vfhq.py vendors/hyperIQA/
cd vendors/hyperIQA/
python iqa_vfhq.py \
    --name train \
    --id_text <id_train.txt> \
    --annotation_folder ../../data/VFHQ/annotations \
    --source_folder ../../data/VFHQ/processed/extracted_cropped_face_results \
    --type train \
    --save_folder ../../data/VFHQ/info/hyperiqa \
    --num_process 8

python iqa_vfhq.py \
    --name test \
    --id_text <id_test.txt> \
    --annotation_folder ../../data/VFHQ/annotations \
    --source_folder ../../data/VFHQ/processed/extracted_cropped_face_results \
    --type test \
    --save_folder ../../data/VFHQ/info/hyperiqa \
    --num_process 8
```

## Dataset neutral attributes

### Brightness

```bash
cd 1_attributes_extraction
python brightness.py \
	<imagedir> \
	<outputpath.csv>
```

# Attributes unification

## CelebV-HQ

- Given you have the following attributes stored:
  - RetinaFace: `./data/CelebV-HQ/info/retina/txt`
  - SynergyNet output: `./data/CelebV-HQ/info/synergy/txt`
  - FacePoseNet output: `./data/CelebV-HQ/info/faceposenet/txt`
  - DirectMHP output: `./data/CelebV-HQ/info/directMHP/{train|test}`
  - IQA output: `./data/CelebV-HQ/info/hyperiqa/txt`

```bash
cd 2_merging/celeb

# Merge Retina with Synergy/ FacePoseNet/ HyperIQA
python cli.py celeb-posemerge-multithread \
	../../data/CelebV-HQ/info/synergy/txt \
	../../data/CelebV-HQ/info/faceposenet/txt \
	../../data/CelebV-HQ/info/hyperiqa/txt \
	../../data/CelebV-HQ/info/retina/txt \
	../../data/CelebV-HQ/merge/1_merge_synergy_anhpose_iqa_retina/all

# Merge above with DirectMHP
python cli.py celeb-directmhp-merge \
	../../data/CelebV-HQ/info/directMHP/all \
	../../data/CelebV-HQ/merge/1_merge_synergy_anhpose_iqa_retina/all \
	../../data/CelebV-HQ/merge/3_merge_all/all

# Binning
python cli.py binning \
	../../data/CelebV-HQ/merge/3_merge_all/all \
	../../data/CelebV-HQ/merge/4_binned/all

cd ..
# Manual step: visualize and fix pose_bin
streamlit run posegrid.py -- ../../data/CelebV-HQ/merge/4_binned/all \
	../../data/CelebV-HQ/processed/zip \
	"softbin" \
	../../data/CelebV-HQ/merge/4_binned_tmp/all
	../../data/CelebV-HQ/merge/4_binned_edited/all

# Merge into a parquet (if wanted)
cd celeb/
python cli.py csvs-to-parquet \
	../../data/CelebV-HQ/merge/4_binned_edited/all \
	../../data/CelebV-HQ/merge/5_parquet/all.parquet
```

## VFHQ

- Given you have the following attributes stored:
  - Groundtruth: `./data/VFHQ/annotations/{train|test}`
  - SynergyNet output: `./data/VFHQ/info/synergy/txt`
  - FacePoseNet output: `./data/VFHQ/info/faceposenet/txt`
  - DirectMHP output: `./data/VFHQ/info/directMHP/{train|test}`
  - IQA output: `./data/VFHQ/info/hyperiqa/txt`
  - zips folder: `./data/VFHQ/zip/{train|test}`

```bash
cd 2_merging/vfhq

# Merge GT with Synergy/ FacePoseNet/ HyperIQA
python cli.py vfhq-posemerge-multithread \
	../../data/VFHQ/info/synergy/txt \
	../../data/VFHQ/info/faceposenet/txt \
	../../data/VFHQ/info/hyperiqa/txt \
	../../data/VFHQ/annotations/{train|test} \
	../../data/VFHQ/merge/1_merge_synergy_anhpose_iqa_anno/{train|test}

# Merge above with DirectMHP
python cli.py vfhq-directmhp-merge \
	../../data/VFHQ/info/directMHP/{train|test} \
	../../data/VFHQ/merge/1_merge_synergy_anhpose_iqa_anno/{train|test} \
	../../data/VFHQ/merge/2_merge_all/{train|test}

# Merge multiple videos into one per id
python cli.py vfhq-combine-multiid-into-one \
	../../data/VFHQ/merge/2_merge_all/{train|test} \
	../../data/VFHQ/merge/3_merge_all_uniqueid/{train|test}


# Binning
python cli.py binning \
	../../data/VFHQ/merge/3_merge_all_uniqueid/{train|test} \
	../../data/VFHQ/merge/4_binned/{train|test}

# Manual step: visualize and fix pose_bin
cd ..
streamlit run posegrid.py -- ../../data/VFHQ/merge/4_binned/{train|test} \
	../../data/VFHQ/processed/zip/{train|test} \
	../../data/CelebV-HQ/merge/4_binned_tmp/all \
	../../data/CelebV-HQ/merge/4_binned_edited/all

# Merge into a parquet (if wanted)
cd vfhq/
python cli.py csvs-to-parquet \
	../../data/VFHQ/merge/4_binned_edited/{train|test} \
	../../data/VFHQ/merge/5_parquet/{train|test}.parquet
```

# Aligning

## StyleGAN

```bash
# VFHQ
cd 2_merging/align
python cli_align.py \
	--bin_folder ../../data/VFHQ/merge/4_binned_edited/ \
	--source_folder ../../data/VFHQ/processed/extracted_hq_results/ \
	--type_folder test \
	--save_folder ../../data/VFHQ/processed/ffhq_aligned \
	--save_bin ../../data/VFHQ/merge/4_binned_ffhq \
	--num_landmarks 5 \
	--size {512|1024} \
	--num_process 1 \
	--dataset_name vfhq

# CelebVHQ:
cd 2_merging/celeb
python celeb_align.py \
	--bin_folder ../../data/CelebV-HQ/merge/4_binned_edited \
	--source_folder ../../data/CelebV-HQ/processed/extracted_hq_results \
	--type_folder all \
	--save_folder ../../data/CelebV-HQ/processed/extracted_cropped_face_results_ffhq \
	--save_bin ../../data/CelebV-HQ/merge/4_binned_ffhq \
	--num_landmarks 5 \
	--size {512|1024} \
	--num_process 1 \
	--dataset_name celeb
```

## EG3D

- Alignment for EG3D is moved below for easier following.

## Face Verification

```bash
cd 3_align_sample
python cli_face_verification.py align \
	<binned_csv_dir> \
	<orginal_frame_dir> \
	<out_dir>
```

# Sampling

## Face Generation

### StyleGAN

```bash
cd 3_align_sample
# for celebv_hq
python cli_sample_gan.py celeb \
	<csv_dir> \
	<aligned_dir> \
	<outdir>
# for vfhq: since we have video_id information, we need to sample it differently from celeb
python cli_sample_gan.py vfhq \
	<parquet_path> \
	<aligned_dir> \
	<outdir>
# in <outdir> will exist a file called `metadata.json` keeping the original filepath and index in the parquet or the original_csv
```

### EG3D

```bash
# from the sampled of stylegan, we retrieve original unaligned images and preprocess it as EG3D
cd 3_align_sample
python cli_get_raws_eg3d.py \
	<metadata.json path_from_aligned> \
	<raw_basedir> \
	<aligned_basedir> \
	<output_basedir>
# align
cd ..
git clone https://github.com/NVlabs/eg3d vendors/eg3d_processing
cp 3_align_sample/eg3d_preprocessing_in_the_wild.py vendors/eg3d_processing/dataset_preprocessing/ffhq/preprocess_in_the_wild.py
# make sure that you install and setup environment to be able to run EG3D/Deep3DReconstruction
cd vendors/eg3d_processing/dataset_preprocessing/ffhq/
python preprocess_in_the_wild.py --indir=<output_basedir>
```

## Face Verification

```bash
cd 3_align_sample
# sample frontal2frontal
python cli_face_verification.py sample-f2f \
	<aligned_dir> \
	<out_dir> \
	--n-pairs=20000
# sample frontal2profile
python cli_face_verification.py sample-f2p \
	<aligned_dir> \
	<out_dir> \
	--n-pairs=20000
# sample profile2profile
python cli_face_verification.py sample-p2p \
	<aligned_dir> \
	<out_dir> \
	--n-pairs=20000
```

- In `<out_dir>`, there should be three text files: `f2f.txt`, `f2p.txt`, `p2p.txt`, corresponding to three testing scenario: frontal2frontal, frontal2profile, profile2profile.

# Building final dataset

In this stage, we will build the final training dataset by either merging with the existing frontal-focused dataset such as: FFHQ, Vox or a standalone one (ControlNet), in case of Face Verification, the dataset is built after the sampling process.

## StyleGAN

- Given two sampled version of VFHQ/ CelebVHQ:
  - `./data/sampled/gan/vfhq`
  - `./data/sampled/gan/celeb`
- Given the preprocessed FFHQ:
  - `./data/gan/compose/ffhq`

```bash
cp -rf ./data/sampled/gan/{vfhq|celeb} ./data/gan/compose
# if building labeled version, skip if not needed
cd 4_building
# Get the FFHQ_Age from https://github.com/royorel/FFHQ-Aging-Dataset
python cli_gan.py <compose_dir> <ffhqAge_csv>
cd ..
# building final zip
cd vendors/stylegan2-ada-pytorch
python dataset_tool.py --source=<compose_dir> --dest=../../data/zips/ffhq_vfhq_celeb_1024x1024.zip
```

## EG3D

- Given two sampled version of VFHQ/ CelebVHQ:
  - `./data/sampled/eg3d/vfhq`
  - `./data/sampled/eg3d/celeb`
  - For each dataset, one can review the aligned dataset and compile list of image to filter into:
  - `./data/sampled/eg3d/vfhq_filter_list.txt`
  - `./data/sampled/eg3d/celeb_filter_list.txt`
- Given the EG3D's preprocessed FFHQ (69957 non-mirrored images/ 139914 mirrored images, 139 subfolders):
  - `./data/eg3d/built`

```bash
# if building labeled version, skipped if not needed
cd 4_building
python cli_eg3d.py \
	../data/eg3d/built \
	../data/sampled/eg3d/celeb \
	../data/sampled/eg3d/celeb/dataset.json \
	../data/sampled/eg3d/celeb_filter_list.txt \
	../data/sampled/eg3d/vfhq \
	../data/sampled/eg3d/vfhq/dataset.json \
	../data/sampled/eg3d/vfhq_filter_list.txt

cd ../data/sampled/eg3d/built
# inspect merged json file if needed, then rename it
mv dataset.json dataset_ffhq.json
mv dataset_merged.json dataset.json
zip -rq0 ../../../zip/eg3d_ffhq_vfhq_celeb_512x512.zip ./*
```

## ControlNet

- Given the sampled dataset for EG3D is located at: `./data/sampled/eg3d/{vfhq|celeb}`

```bash
cd 1_attributes_extraction
# Generate face attributes
python generate_face_attributes.py \
	./data/sampled/eg3d \
	./data/sampled/eg3d_aux/face_attributes

# Generate captioning from BLIP
python generate_blip.py \
	./data/sampled/eg3d \
	./data/sampled/eg3d_aux/blip.json

# Generate prompt
python generate_prompt.py \
	./data/sampled/eg3d/aux/face_attributes \
	./data/sampled/eg3d_aux/prompt.json

# Generate 478 landmarks
python generate_face_pose_landmarks_only.py extract \
	./data/sampled/eg3d
	./data/sampled/eg3d_aux/landmarks/mediapipe

# Generate condition image
python generate_face_pose_landmarks_only.py draw_lmk_mediapipe \
	./data/sampled/eg3d_aux/landmarks/mediapipe \
	./data/sampled/eg3d_aux/landmarks_img/mediapipe

# build controlnet dataset
mkdir -p ./data/sampled/controlnet
cp -rf ./data/sampled/eg3d_aux/landmarks_img/mediapipe ./data/sampled/controlnet/condition_images
cp -rf ./data/sampled/eg3d ./data/sampled/controlnet/target_images
cp -rf ./data/sampled/eg3d_aux/prompt.json ./data/sampled/controlnet
```

## Face Reenactment

### Vox Preprocessing

```bash
cd 4_building/reenactment
python preprocess_vox.py \
    --name demo \
    --dataset_name celeb \
	--annotation ../../data/CelebV-HQ/merge/1_merge_synergy_anhpose_iqa_retina \
	--source_folder ../../data/CelebV-HQ/processed/extracted_hq_results \
    --bbox_folder ../../data/CelebV-HQ/processed/vox_bbox \
	--type all \
	--save_folder ../../data/CelebV-HQ/processed/extracted_cropped_face_results_vox \
	--workers 1

python preprocess_vox.py \
    --name test \
    --dataset_name vfhq \
	--annotation ../../data/VFHQ/merge/2_merge_all \
	--source_folder ../../data/VFHQ/processed/extracted_hq_results \
    --bbox_folder ../../data/VFHQ/processed/vox_bbox \
	--type test \
	--save_folder ../../data/VFHQ/processed/extracted_cropped_face_results_vox \
	--workers 1

python preprocess_vox.py \
    --name train \
    --dataset_name vfhq \
	--annotation ../../data/VFHQ/merge/2_merge_all \
	--source_folder ../../data/VFHQ/processed/extracted_hq_results \
    --bbox_folder ../../data/VFHQ/processed/vox_bbox \
	--type train \
	--save_folder ../../data/VFHQ/processed/extracted_cropped_face_results_vox \
	--workers 5
```

### LightFace for Celeb

```bash
cd 4_building/reenactment
python verify.py \
    --bin_folder ../../data/CelebV-HQ/merge/4_binned_edited/ \
	--source_folder ../../data/CelebV-HQ/processed/extracted_hq_results \
	--type_folder all \
	--json_folder ../../data/CelebV-HQ/info/lightface \
	--num_process 5
```

### Reorganize CelebV-Reenactment data

```bash
cd 4_building/reenactment
python reformat.py \
	--source_folder ../../data/CelebV-HQ/processed/extracted_cropped_face_results_vox \
	--save_folder ../../data/CelebV-HQ/processed/extracted_cropped_face_results_vox_final \
	--type_folder all \
	--json_folder ../../data/CelebV-HQ/info/lightface \
	--num_process 5
```

### Get text data file for reenactment data

The text file will contain: path_to_image and category (frontal/extreme)

```bash
cd 4_building/reenactment
python vfhq_txt.py \
	--source_folder ../../data/VFHQ/processed/extracted_cropped_face_results_vox \
	--bin_folder ./../data/VFHQ/merge/4_binned \
	--text_path ../../data/VFHQ/info/all_instance.txt \
	--json_path ../../data/VFHQ/info/all_instance.json

python celeb_txt.py \
	--type_folder all \
	--source_folder ../../data/CelebV-HQ/processed/extracted_cropped_face_results_vox_final \
	--bin_folder ./../data/CelebV-HQ/merge/4_binned_edited \
	--type_folder all \
	--text_path ../../data/CelebV-HQ/info/all_instance.txt \
	--json_path ../../data/VFHQ/info/all_instance.json

python create_train_test.py \
	--dataset_name vfhq \
	--test_pct 0.25 \
	--json_path ../../data/VFHQ/info/all_instance.json \
	--text_path ../../data/VFHQ/info/all_instance.txt \
	--annotation_folder ../../data/VFHQ/info/reenactment

python create_train_test.py \
	--dataset_name celeb \
	--test_pct 0.25 \
	--json_path ../../data/CelebV-HQ/info/all_instance.json \
	--text_path ../../data/CelebV-HQ/info/all_instance.txt \
	--annotation_folder ../../data/CelebV-HQ/info/reenactment
```

# Training

```bash
mkdir -p training_code
```

## Face synthesis

### StyleGAN

```bash
git clone <hidden_for_anonymous> training_code/efhq_stylegan
cd training_code/efhq_stylegan
# envsetup as StyleGAN2-ADA-Pytorch repo, then LET'S TRAIN!
python train.py --outdir=./training-runs --data=../../data/zips/ffhq_vfhq_celeb_1024x1024.zip --gpus=8 --cfg=paper1024 --mirror=1
```

### EG3D

```bash
git clone <hidden_for_anonymous> training_code/efhq_eg3d
cd training_code/efhq_eg3d
# envsetup as EG3D repo, then LET'S TRAIN!
# phase 1
python train.py --outdir=./training-runs --cfg=ffhq --data=../../data/zips/eg3d_ffhq_vfhq_celeb_512x512.zip \
  --gpus=8 --batch=32 --gamma=5 --gen_pose_cond=True --metrics=fid2k_full
# phase 2
python train.py --outdir=./training-runs --cfg=ffhq --data=../../data/zips/eg3d_ffhq_vfhq_celeb_512x512.zip \
  --resume=<...>/network-snapshot-025000.pkl \
  --gpus=8 --batch=32 --gamma=1 --gen_pose_cond=True --neural_rendering_resolution_final=128 --metrics=fid2k_full
```

### ControlNet

```bash
git clone https://github.com/lllyasviel/ControlNet training_code/efhq_controlnet
cp -f 5_train_eval/controlnet/* training_code/efhq_controlnet
cd training_code/efhq_controlnet
# envsetup as StyleGAN2-ADA-Pytorch repo, then LET'S TRAIN!
python tutorial_train.py
```

## Face Reenactment

For instructions to download VoxCeleb1, please refer to this repo: https://github.com/AliaksandrSiarohin/video-preprocessing

### TPS

For config files, please pay attention to the image and annotation paths for each dataset.

```bash
git clone https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model.git training_code/tps
cp -rf 5_train_eval/tps/* training_code/tps
cd training_code/tps
# envsetup as TPS repo, then LET'S TRAIN!
CUDA_VISIBLE_DEVICES=0,1 python run.py --config config/ext-256.yaml --device_ids 0,1
CUDA_VISIBLE_DEVICES=0 python run.py --mode train_avd --checkpoint '{checkpoint_folder}/checkpoint.pth.tar' --config config/ext-256.yaml
```

### LIA

```bash
git clone https://github.com/wyhsirius/LIA.git training_code/lia
cp -rf 5_train_eval/lia/* training_code/lia
cd training_code/lia
# envsetup as LIA repo, then LET'S TRAIN!
python train.py --dataset ext --exp_path exps/ --exp_name ext_bs32_lr0.002
python train.py --dataset vox --exp_path exps/ --exp_name vox_bs32_lr0.002
```

# Evaluation

## Face Synthesize

### StyleGAN

```bash
cd training/efhq_stylegan
# evaluate label as dataset
python calc_metrics.py --metrics=fid50k_full --network=<model_path> --data=../../data/zips/ffhq_vfhq_celeb_1024x1024.zip
# evaluate frontal label only
python calc_metrics.py --metrics=fid_50kfull --network=<model_path> --eval_type=frontal --data=<ffhq_zip_path>
# evaluate lpff distribution
python calc_metrics.py --metrics=fid_50kfull --network=<model_path> --eval_type=lpff --data=<lpff_zip_path>
```

### EG3D

```bash
cd training/efhq_eg3d
python calc_metrics.py --metrics=fid50k_full \
    --network=<data_path>
```

### ControlNet

```bash
git clone https://github.com/lllyasviel/ControlNet training_code/efhq_controlnet_v1.1
cp 1_attributes_extraction/path_utils.py 5_train_eval/controlnet/cli_openpose.py 5_train_eval/controlnet/crop_eyepatch.py 5_train_eval/controlnet/eval.py training_code/efhq_controlnet_v1
cd training_code/efhq_controlnet_v1.1
# generated samples
python cli_openpose.py \
	<ckpt_path> \
	../../data/sampled/controlnet/condition_images \
	<prompt> \
	<outdir> \
	--no-need-detect

# generated 478 keypoints
cd ../1_attributes_extraction
python generate_face_pose_landmarks_only.py extract \
	<outdir>
	<lmk_outdir>

cd ../5_train_eval/controlnet
# crop eyepatch
python crop_eyepatch.py \
	<outdir> \
	<lmk_outdir> \
	<eyepatch_outdir>

# Eval NME
python eval.py \
	../../data/sampled/eg3d_aux/landmarks/mediapipe \
	<lmk_outdir>

# For FID evaluation, we utilize https://github.com/mseitzer/pytorch-fid
```

## Face Reenactment

### TPS

```bash
cd training_code/tps
CUDA_VISIBLE_DEVICES=0 python run.py --mode reconstruction --config config/test/ext-256.yaml --checkpoint '{checkpoint_folder}/checkpoint.pth.tar'
CUDA_VISIBLE_DEVICES=0 python run.py --mode reconstruction --config config/test/vox-256.yaml --checkpoint '{checkpoint_folder}/checkpoint.pth.tar'
CUDA_VISIBLE_DEVICES=0 python run.py --mode reconstruction --config config/test/ext-256_cross.yaml --checkpoint '{checkpoint_folder}/checkpoint.pth.tar'
```

The reconstruction subfolder will be created in {checkpoint_folder}. The generated video will be stored to this folder, also generated videos will be stored in png subfolder in loss-less '.png' format for evaluation. To compute metrics, follow instructions from [pose-evaluation](https://github.com/AliaksandrSiarohin/pose-evaluation).

### LIA

```bash
cd training_code/tps
python evaluation.py --dataset ext --ckpt_path '{checkpoint_folder}/checkpoint.pt' --save_path <SAVE_PATH>
python evaluation.py --dataset ext_cross --ckpt_path '{checkpoint_folder}/checkpoint.pt' --save_path <SAVE_PATH>
python evaluation.py --dataset vox --ckpt_path '{checkpoint_folder}/checkpoint.pt' --save_path <SAVE_PATH>
```

The reconstruction subfolder will be created in \<SAVE_PATH>. To compute metrics, follow instructions from [pose-evaluation](https://github.com/AliaksandrSiarohin/pose-evaluation).

## Face Verification

```bash
git clone https://github.com/deepinsight/insightface vendors/insightface
cp 5_train_eval/facevec/inference.py 3_align_sample/cli_face_verification.py  3_align_sample/face_verification_evaltoolkit.py  vendors/insightface/recognition/arcface_torch
cd vendors/insightface/recognition/arcface_torch
# Setup env and download corresponding weights from the author github
# Example for R100
# Get features
python inference.py --network=r100 --weight=./pretrained/glint360k_cosface_r100_fp16_0.1.pth --dir=<facevec_aligned_sample_outdir> --bs=256 --out=<facevec_feat_outdir>

# Evaluate
python cli_face_verification.py eval \
	<facevec_feat_outdir> \
	{f2f|f2p|p2p}.txt
```
