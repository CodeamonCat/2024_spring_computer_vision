# Real-Time Monitoring of Door Status in Public Transit Systems

### Introduction

This is the final project competition of the Computer Vision Course (Spring 2024, National Taiwan University) sponsored by Vivotek.

### Getting Started

This folder contains the implementation of training a fine-tuned Swin Transformer V2 for image classification and the pre-trained models and weights.

#### Install

- Create a conda virtual environment and activate it:

```bash
conda create -n swin python=3.10 -y
conda activate swin
```

- Install requirements:

```bash
pip install -r requirements
```

#### Data preparation

To predict the results of videos, place the target videos into the `Tests` folder at the same directory.

### Generate prediction

```bash
python generate_output.py
```

### Train our fine-tuned model

:warning: Once a model is trained, it will replace the pre-trained model.

```bash
python train.py && python generate_output.py
```

### Reference

#### Citing Swin Transformer

```
@inproceedings{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

#### Citing Swin Transformer V2

```
@inproceedings{liu2021swinv2,
  title={Swin Transformer V2: Scaling Up Capacity and Resolution},
  author={Ze Liu and Han Hu and Yutong Lin and Zhuliang Yao and Zhenda Xie and Yixuan Wei and Jia Ning and Yue Cao and Zheng Zhang and Li Dong and Furu Wei and Baining Guo},
  booktitle={International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```

#### Citing Hugging Face Transformers Library

```
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```
