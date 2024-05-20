# EFHQ: Multi-purpose ExtremePose-Face-HQ dataset
[![Project](https://img.shields.io/badge/Website-EFHQ-purple)](https://bomcon123456.github.io/efhq/)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2312.17205)

#### Table of content
1. [Abstract](#abstract)
1. [Usage](#usage)
1. [Citation](#citation)
1. [Agreement](#agreement)


## Abstract
> The existing facial datasets, while having plentiful images at near frontal views, lack images with extreme head poses, leading to the downgraded performance of deep learning models when dealing with profile or pitched faces. This work aims to address this gap by introducing a novel dataset named Extreme Pose Face High-Quality Dataset (EFHQ), which includes a maximum of 450k high-quality images of faces at extreme poses. To produce such a massive dataset, we utilize a novel and meticulous dataset processing pipeline to curate two publicly available datasets, VFHQ and CelebV-HQ, which contain many high-resolution face videos captured in various settings. Our dataset can complement existing datasets on various facial-related tasks, such as facial synthesis with 2D/3D-aware GAN, diffusion-based text-to-image face generation, and face reenactment. Specifically, training with EFHQ helps models generalize well across diverse poses, significantly improving performance in scenarios involving extreme views, confirmed by extensive experiments. Additionally, we utilize EFHQ to define a challenging cross-view face verification benchmark, in which the performance of SOTA face recognition models drops 5-37\% compared to frontal-to-frontal scenarios, aiming to stimulate studies on face recognition under severe pose conditions in the wild.

<p align="center">	
<img width="750" alt="entity types" src="./assets/teaser.png">
</p>

## Usage
For a full reproduce, please follow the detailed guideline over [here](./docs/detail.MD)

We provide links to [pretrained-models](https://huggingface.co/EFHQ/efhq_weights/tree/main) and [metadata](https://drive.google.com/drive/folders/1S7kmzsTsdrdyL7tl-p9gfnolcR4Qo9LE?usp=drive_link) files for each task. In accordance with the original datasets, we supply metadata only for the frames that are included in the final dataset. Please download the video first, and then use our metadata to construct the final dataset.

## Citation
If you find our paper is useful for your work, consider citing:

    @inproceedings{dao2024efhq,
      title={EFHQ: Multi-purpose ExtremePose-Face-HQ dataset}, 
      author={Trung Tuan Dao and Duc Hong Vu and Cuong Pham and Anh Tran},
      year={2024},
	  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    }
Please **CITE** our paper whenever this repository is used to help produce published results or incorporated into other software
	
## Agreement
By downloading the EFHQ dataset, USER agrees:

- to use EFHQ for research or educational purposes only.
- to **not** distribute EFHQ or part of EFHQ in any original or modified form.
- and to cite our paper above whenever EFHQ is employed to help produce published results.

## Copyright (c) 2024 VinAI Research

	THE DATA IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE DATA OR THE USE OR OTHER DEALINGS IN THE
	DATA.

