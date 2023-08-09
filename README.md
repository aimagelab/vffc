# vffc

This repository contains the reference code and dataset for the paper [Volumetric Fast Fourier Convolution for Detecting Ink on the Carbonized Herculaneum Papyr](https://arxiv.org/abs/).
If you find it useful, please cite it as:
```
@inproceedings{quattrini2023volumetric,
  title={{Volumetric Fast Fourier Convolution for Detecting Ink on the Carbonized Herculaneum Papyr}},
  author={Quattrini, Fabio and Pippi, Vittorio and Cascianelli, Silvia and Cucchiara, Rita},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  year={2023}
}
```

## Setup
We provide the [checkpoint](https://github.com/aimagelab/vffc_anonym/releases/download/Latest/8eb9.pth) for the model:

## Setup
To run this project, we used `python 3.10.12` and `pytorch 2.0`. We provide the environment packages in vffc.yml 
```bash
conda create --name vffc
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r vffc.yml
```

## Inference
To run the model on a folder with images, run the following command:
```
python inference.py --checkpoint_path <path to checkpoint> --papyrus_path <path to the test papyrus folder> 
--output_path <path to the output folder> 
```
Optional arguments are:
``` 
--patch_size <patch size>   (default=256)
--test_stride <test stride> (default=32)
--batch_size <batch size>   (default=4)
--threshold <threshold>     (default=0.5)
--z_start <z_start>         (default=24)
--z_size <z_size>           (default=16)
--stride <stride>           (default=64)
```

The papyrus folder should have the following structure:

```
papyrus_id
├── mask.png
├── surface_volume/
│   ├── 00.tif
│   ├── ...
│   ├── 64.tif
```

## Training
To train the model, run the following command:
```
python train.py --train_data_path <path to the train folder> --outputs_path <path to the output folder>
```

The train folder should have the following structure:

```
papyrus_1
├── mask.png
├── inklabels.png
├── surface_volume/
│   ├── 00.tif
│   ├── ...
│   ├── 64.tif
papyrus_2
├── mask.png
├── inklabels.png
├── surface_volume/
│   ├── 00.tif
│   ├── ...
│   ├── 64.tif
papyrus_3
├── mask.png
├── inklabels.png
├── surface_volume/
│   ├── 00.tif
│   ├── ...
│   ├── 64.tif
```

The main optional arguments are:
``` 
--patch_size <patch size>           (default=256)
--z_start <z_start>                 (default=24)
--z_size <z_size>                   (default=16)
--batch_size_train <batch size>     (default=4)
--batch_size_eval <batch size>      (default=4)
--threshold <threshold>             (default=0.5)
--train_stride <train_stride>       (default=64)
--eval_stride <eval_stride>         (default=64)
--test_stride <test_stride>         (default=64)
--enable_train_augmentations        (default=True)
--loss <loss>                       (default='bce_dice')
```




