# BAE-ViT: An Efficient Multimodal Vision Transformer for Bone Age Estimation
**This is an official implementation of BAE-ViT**

BAE-ViT is a novel vision transformer model optimized for bone age estimation (BAE) that efficiently integrates both image and gender information. Utilizing tokenization techniques, the model allows for intricate interactions between visual and non-visual data, thereby enhancing performance, robustness, and interpretability. Trained on a comprehensive dataset from the 2017 RSNA Pediatric Bone Age Machine Learning Challenge, BAE-ViT demonstrates competitive performance against existing models and exhibits robustness to image distortions. Statistical analyses further validate the model's efficacy, confirming its alignment with ground truth labels. The study underscores the potential of vision transformers as a superior alternative to traditional CNNs for multimodal data integration in medical imaging scenarios.

**This repository is credited to [Jinnian Zhang (DominickZhang)](https://github.com/DominickZhang).**

## Build Environment
- Create a conda virtual environment and activate it:

```bash
conda create -n baevit python=3.7 -y
conda activate baevit
```

- Install `CUDA==10.1` with `cudnn7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch==1.7.1` and `torchvision==0.8.2` with `CUDA==10.1`:

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```

- Install other requirements:
```bash
pip install -r requirements.txt
```

## Prepare Data
The official RSNA dataset can be downloaded [here](https://www.kaggle.com/datasets/kmader/rsna-bone-age/). There are 12611 training images, 1425 validation images, and 200 test images.

The folder structure is shown below:
```
dataset
├── rsna
│   ├── rsna-train.csv
│   ├── rsna-validation.csv
│   ├── rsna-test.csv
│   ├── rsna-train
│   │   ├── 1377.png
│   │   ├── 1378.png
...
│   ├── rsna-validation
│   │   ├── 2001.png
│   │   ├── 2002.png
...
│   ├── rsna-test
│   │   ├── 4360.png
│   │   ├── 4361.png
```

**Note:** In this demo, to avoid the size of this project becoming too large, we simply duplicate the test data to create the training and validation data.

## Model Checkpoint
The pretrained model checkpoint (`baevit-ckpt_epoch_299.pth`) is available for download from [Google Drive](https://drive.google.com/file/d/1le9yPFakua-wLsI-M6gfdoeD1Pz70AAO/view?usp=share_link).

## Evaluate
Run the following command for center-crop test:
```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12413 main.py --cfg configs/baevit.yaml --data-path dataset/rsna --batch-size 32 --accumulation-steps 1 --output output/rsna --tag inference --criterion l1 --random_seed 0 --eval --test_only --resume output/baevit-ckpt_epoch_299.pth
```

Run the following command for multi-crop test:
```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12413 main.py --cfg configs/baevit.yaml --data-path dataset/rsna --batch-size 32 --accumulation-steps 1 --output output/rsna --tag inference --criterion l1 --random_seed 0 --eval --test_only --multicrop_test --resume output/baevit-ckpt_epoch_299.pth
```
