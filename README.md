# TransDeepLab

The code for "_TransDeepLab: Convolution-Free Transformer-based DeepLab v3+ for Medical Image Segmentation_".

![Proposed Model](./images/proposed_model.png)

---

## Updates
- July 18, 2022: First release for 

## Setting up and Training

- We use the code base from [Swin-Unet GitHub repo](https://github.com/HuCaoFighting/Swin-Unet) as our starting point.

- In order to run the code and experiments, you need to first install the dependencies and then download and move the data to the right place. 

- I have put the required instructions for doing the above steps in the `./setup.sh` file in the repo. `cd` to this repo directory and then run it to install dependencies and download and move data to the right dir.

    -  For the _Synapse_ dataset, we used the data provided by [TransUnet](https://github.com/Beckschen/TransUNet)'s authors.
    - For _ISIC 2017-18_ datasets, we used the ISIC Challenge datasets [link](https://challenge.isic-archive.com/data/).
    - For the _PH$^2$_ dataset, we used this [link](https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar).

- Then you need to run the `train.py` file with the appropriate arguments to run your experiment. I'll explain this in further detail in the next section; let's just see an example for now for the _Synapse_ dataset:

```bash
python train.py --config_file 'swin_224_7_{# of SSPP}level' --dataset Synapse --root_path './data' --max_epochs 200 --output_dir '.'  --img_size 224 --base_lr 0.05 --batch_size 24
```
---

## Config files and hyperparameters

- All the hyperparameters related to building different models are in separate files in `./model/configs` directory.

- For each experiment, you need to make your desired `config_name.py` file and put it in the `model/configs` dir and then enter the file name (without `.py` suffix) in the `python train.py` command you saw in the previous section after `--config_file` arg.

- For the rest of the hyperparameters like `batch_size`, `max_epochs`, `base_lr`, ... look at the `Swin-Unet` or the code here to see what you can change and how to do so.

---
## Test
- The model can be tested with the following command using `test.py` file. During training, model checkpoints will be saved to disk with the following format: `output_dir/{config_file_name}_epoch_{epoch_num}.pth`.

- It takes the checkpoint (model weight file) name as an input argument and loads the appropriate config file from the configs dir.

- Other arguments and flags can be given to the `test.py` file if some settings need to be modified but `--ckpt_path` and `--config_file` are the only required arguments.

- Trained weights for our best-reported results in the paper are easily accessible from [this link](https://drive.google.com/drive/folders/17AYvKNYIHYvbhkOEE8VRO5vbADNYQEVG?usp=sharing), where you could download it as a sole folder via [gdown](https://github.com/wkentaro/gdown) or setting specific links listed under the below table.

| Model setting name | Pre-trained weights |
| --- | --- |
|SSPP Level 1 | [link](https://drive.google.com/file/d/1gjYUEi3fw90IgenmlLzms2qx8f_-WXJe/view?usp=sharing)|
| SSPP Level 2 | [link](https://drive.google.com/file/d/1UuZrFcZNRMAc6c_xiNMK471n1d1r4ows/view?usp=sharing) |
| SSPP Level 3 | [link](https://drive.google.com/file/d/111KqDd0SVKKJtLnQlTDaJi03WO9CsZew/view?usp=sharing) |
| SSPP Level 4 | [link](https://drive.google.com/file/d/1015liUD9gz6sygtvMH6oGb0oHqODFsoW/view?usp=sharing) |

|**Setting**| DSC$\uparrow$   | HD$\downarrow$ | Aorta | Gallbladder | Kidney(L) | Kidney(R)| Liver | Pancreas| Spleen | Stomach |
| ------------------------------------------------------------------------------ |:----------------:|:---------------:|:-------:|:-----------:|:---------:|:----------------:|:-------:|:----------------:|:-------:|:-------:|
| **CNN as Encoder**                                                             | $75.89$          | $28.87$         | $85.03$ | $65.17$     | $80.18$   | $76.38$          | $90.49$ | $57.29$          | $85.68$ | $69.93$ |
| **Basic Scale Fusion**                                                         | $79.16$          | $22.14$         | $85.44$ | $68.05$     | $82.77$   | $80.79$          | $93.80$ | $58.74$          | $87.78$ | $75.96$ |
| **SSPP Level 1**                                                               | $79.01$          | $26.63$         | $85.61$ | $68.47$     | $82.43$   | $78.02$          | $94.19$ | $58.52$          | $88.34$ | $76.46$ |
| **SSPP Level 2**                                                               | $80.16$          | $21.25$         | $86.04$ | $69.16$     | $84.08$   | $79.88$          | $93.53$ | $61.19$          | $89.00$ | $78.40$ |
| **SSPP Level 3**                                                               | $79.87$          | $18.93$         | $86.34$ | $66.41$     | $84.13$   | $82.40$          | $93.73$ | $59.28$          | $89.66$ | $76.99$ |
| **SSPP Level 4**                                                               | $79.85$          | $25.69$         | $85.64$ | $69.36$     | $82.93$   | $81.25$          | $93.09$ | $63.18$          | $87.80$ | $75.56$ |


(should be checked)
A Look at the Number of Parameters

| Model | # Encoder Parameters | # ASPP Parameters | # Decoder Parameters | # Total |
| --- | ----------- | --- | --- | --- |
| Original DeepLab (Xception-512) | 37.86 | 15.53 | 1.30  | 54.70 |
| Original Swin-Unet (224-7) | 27.51 | None | 13.87 | 41.38 |
| Our SwinDeepLab (swin_224_7_18) **(final version)** | 12.15 | 9.20 | 3.49 | **24.85** |
| Our SwinDeepLab (swin_224_7_{0,1}) | 12.15 | 8.98 | 3.49 | 24.63 |
| Our SwinDeepLab (swin_224_7_2) | 12.15 | 23.20 | 3.49 | 38.64 |
| Our SwinDeepLab (swin_224_7_{3..12}) | 12.15 | 8.98 | 3.49 | 24.63 |
| Our SwinDeepLab (swin_224_7_13) | 12.15 | 8.98 | 4.39 | 25.53 |
| Our SwinDeepLab (swin_224_7_14) | 12.15 | 8.98 | 7.06 | 28.20 |
| Our SwinDeepLab (resnet_224_0) | 8.97 | 9.02 | 6.19 | 24.19 |

---
## References
- [TransUNet](https://github.com/Beckschen/TransUNet)
- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
---
## Citation
```

```
