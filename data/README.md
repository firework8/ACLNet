# Pretrained Models

## Introduction

We present the detailed performance on various datasets and modalities. 

All the checkpoints and training logs are provided in the [Google Drive](https://drive.google.com/drive/folders/1UheUd00odocDjtQtWY9SNIbFDuA9qs1a?usp=sharing). We sincerely hope that this repo could be helpful for your research.

## Experimental Results

The results for pretrained models are displayed below. The detailed comparisons for various datasets are provided in `{dataset}_ensemble.py`.


| Modality | NTU 60 X-Sub | NTU 60 X-View | NTU 120 X-Sub | NTU 120 X-Set |
|:---:|:---:|:---:|:---:|:---:|
| Joint| [91.44](https://drive.google.com/drive/folders/1OwFemGSvHSFJp5qJH75qAWvhsiDiENjK?usp=sharing) | [96.38](https://drive.google.com/drive/folders/19OJ-5b07bK2vj4hFqhYelwqp7zZAA7ri?usp=sharing) | [85.39](https://drive.google.com/drive/folders/1W-igIeCeHMP4FRoF_00yfYfRfN6pGLrl?usp=sharing) | [88.33](https://drive.google.com/drive/folders/11dqy4ZCWee8nww21XKWF5HSNu2XZODVn?usp=sharing) |
| Bone | [91.90](https://drive.google.com/drive/folders/1cin2cW8zqdq3wMu1zt-V89j9u2yFA-g7?usp=sharing) | [95.97](https://drive.google.com/drive/folders/1fyYN2csJZNGexAd8Wfrx9MKJuMxOLrUf?usp=sharing) | [88.63](https://drive.google.com/drive/folders/1xLEdYlcNj0wNtVEXMbxcvVeHdj9Z8yuZ?usp=sharing) | [90.11](https://drive.google.com/drive/folders/1j4OIdepjLGPw6lvc8R2LDVUqbBHjQchv?usp=sharing) |
| 2-ensemble | [93.07](https://drive.google.com/drive/folders/1O6JPZbUOo88GQsP1PfIknp_MghcCHwjy?usp=sharing) | [97.26](https://drive.google.com/drive/folders/1C-RCu7WsMR4YUar16PHyBkTVkaBosDxb?usp=sharing) | [89.62](https://drive.google.com/drive/folders/1rdsO26M4sEwIOwVqg4e9TuwuWh9rLaRx?usp=sharing) | [91.38](https://drive.google.com/drive/folders/1NoSnXSNm6yRiMM4O5m4fgIO-eqzPGvoB?usp=sharing) |
| 4-ensemble | [93.43](https://drive.google.com/drive/folders/1O6JPZbUOo88GQsP1PfIknp_MghcCHwjy?usp=sharing) | [97.63](https://drive.google.com/drive/folders/1C-RCu7WsMR4YUar16PHyBkTVkaBosDxb?usp=sharing) | [90.35](https://drive.google.com/drive/folders/1rdsO26M4sEwIOwVqg4e9TuwuWh9rLaRx?usp=sharing) | [92.00](https://drive.google.com/drive/folders/1NoSnXSNm6yRiMM4O5m4fgIO-eqzPGvoB?usp=sharing) |
| 6-ensemble | [**93.58**](https://drive.google.com/drive/folders/1O6JPZbUOo88GQsP1PfIknp_MghcCHwjy?usp=sharing) | [**97.74**](https://drive.google.com/drive/folders/1C-RCu7WsMR4YUar16PHyBkTVkaBosDxb?usp=sharing) | [**90.72**](https://drive.google.com/drive/folders/1rdsO26M4sEwIOwVqg4e9TuwuWh9rLaRx?usp=sharing) | [**92.29**](https://drive.google.com/drive/folders/1NoSnXSNm6yRiMM4O5m4fgIO-eqzPGvoB?usp=sharing) |


| Modality | Kinetics-Skeleton | PKU-MMD X-Sub | PKU-MMD X-View | FineGYM |
|:---:|:---:|:---:|:---:|:---:|
| Joint| [48.37](https://drive.google.com/drive/folders/1K7TMfMqyJSy-IHXZjTxhil9Wvvt68lNP?usp=sharing) | [96.04](https://drive.google.com/drive/folders/1AKTTTpTlrw94I--a22tKJF5epNMsa1sN?usp=sharing) | [98.25](https://drive.google.com/drive/folders/1_UxcUy1XIHJnG1JkCF_wfpGdoW6pWnrA?usp=sharing) | [93.31](https://drive.google.com/drive/folders/1eJDVpJ2yPI100VYquvc2yjCjrg1ywcg7?usp=sharing) |
| Bone | [47.22](https://drive.google.com/drive/folders/1wpDKVDTwRM6tXiMHgT9mJbi7Yo5IaS_u?usp=sharing) | [96.67](https://drive.google.com/drive/folders/1khBnzVw5vZXnwIpEuK0ACHSb25mOuTsH?usp=sharing) | [98.01](https://drive.google.com/drive/folders/1E2ueZt15EDHLFpwZC0zfjaVGBC8uIzbV?usp=sharing) | [94.81](https://drive.google.com/drive/folders/1IonK-TPwcN0bQbO8qJ3c0DVp0KBVXENt?usp=sharing) |
| 2-ensemble | [50.35](https://drive.google.com/drive/folders/1KxyTzjcKBGWGun2Gl0Pm0Q2U6MVLL_kk?usp=sharing) | [96.71](https://drive.google.com/drive/folders/19MXseVqbet5CRezQplt9dmq8Q-lAPQlz?usp=sharing) | [98.47](https://drive.google.com/drive/folders/1m7ENlvhd390YOe_GY2n8gQOqU6fKt_GN?usp=sharing) | [95.48](https://drive.google.com/drive/folders/1koTnkXWSTOzlpoQJxjkg1SijenuECahK?usp=sharing) |
| 4-ensemble | [51.50](https://drive.google.com/drive/folders/1KxyTzjcKBGWGun2Gl0Pm0Q2U6MVLL_kk?usp=sharing) | [97.00](https://drive.google.com/drive/folders/19MXseVqbet5CRezQplt9dmq8Q-lAPQlz?usp=sharing) | [98.59](https://drive.google.com/drive/folders/1m7ENlvhd390YOe_GY2n8gQOqU6fKt_GN?usp=sharing) | [95.80](https://drive.google.com/drive/folders/1koTnkXWSTOzlpoQJxjkg1SijenuECahK?usp=sharing) |
| 6-ensemble | [**52.13**](https://drive.google.com/drive/folders/1KxyTzjcKBGWGun2Gl0Pm0Q2U6MVLL_kk?usp=sharing) | [**97.06**](https://drive.google.com/drive/folders/19MXseVqbet5CRezQplt9dmq8Q-lAPQlz?usp=sharing) | [**98.65**](https://drive.google.com/drive/folders/1m7ENlvhd390YOe_GY2n8gQOqU6fKt_GN?usp=sharing) | [**96.01**](https://drive.google.com/drive/folders/1koTnkXWSTOzlpoQJxjkg1SijenuECahK?usp=sharing) |
