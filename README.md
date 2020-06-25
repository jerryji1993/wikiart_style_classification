# Wikiart style classification
This repository contains analysis done in study [Zhu, Ji, Zhang, Xu, Zhou, Chan, Arxiv, 2019](https://arxiv.org/abs/1911.10091). The study uses several state-of-the-art CNN architectures to explore how a machine may help perceive and quantify art styles, and explores (1) How accurately can a machine classify art styles? (2) What may be the underlying relationships among different styles and artists?

## Getting Started

### Prerequisites
* Python 3.7
* Tensorflow (2.0.0)
* Numpy (1.18.1)
* Pandas (1.0.3)
* Matplotlib (3.1.3)
* Scikit-learn (0.22.1)
* Pillow (7.1.2)
* Download data from https://www.wikiart.org/ with relabeled train.csv and test.csv

### Model training
First, put all downloaded Wikiart images at ```data/ALL```. To train specific CNN models with specified hyperparameters, check out table below or ```python train_models.py --help``` for complete list of arguments.

Arguments | Description
--------------|---------------------------------------------------------
--input_dir | Input directory (str, required)
--output_dir | Output directory (str, required)
--model_type | Type of model used (str, required)
--batch_size | Batch size used (int, default = 150)
--learning_rate | Learning rate for training the model (float, default = 1e-5)
--image_size | Image size of model (int, default = 300)
--num_epochs | Number of epochs to train (int, default = 100)
--subset | Length to subset a portion of total data for training (int, default = 1000000)
--add_FC | Add fully connected layer before softmax for obtaining features? (store_true, default = False)
--n_threads | Number of threads (int, default = 1)

Choices of CNN architectures:
* VGG16
* ResNet152
* ResNet152V2
* ResNeXt101
* InceptionV3
* InceptionResNetV2
* EfficientNet (B7)
* NASNet (Large)

### Grad-CAM visualization
We also implemented Grad-CAM for direct visualzation of where the model is paying attention to when making classification decision. Once the model is trained, one can run ```python vis_cam.py``` for generation of class activation maps. Table below lists some arguments to specify for generating the maps:
