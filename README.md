# DeepLearning Paper List
A list of paper that i read

## Contents
- [Computer-Vision](#Computer-Vision)
  - [Self-Supervised-Learning](#Self-Supervised-Learning)
  - [ViewPoint-Estimation](#ViewPoint-Estimation)
  - [Image-Synthesis,Image-Transfer](#Image-Synthesis,Image-Transfer)
  - [Image-Augmentation](#Image-Augmentation)
  - [Dataset](#Dataset) 
- [Natural-Language-Processing](#Natural-Language-Processing)
  - [Audio-Synthesis](#Audio-Synthesis)
- [Data-Privacy](#Data-Privacy) 
- [Medical-application](#Medical-application)

## Computer-Vision
### Self-Supervised Learning

Title | Contributions | Code | review |
--- | --- | --- | --- |
Diverse Image Generation via Self-Conditioned GANs(CVPR 2020)[[pdf]](https://arxiv.org/abs/2006.10728) | <ul><li> Self-Conditional GAN by clustering </il><li> Computing Dataset partition by clustering | [code](https://github.com/stevliu/self-conditioned-gan) | [Yes](https://medium.com/analytics-vidhya/paper-review-diverse-image-generation-via-self-conditioned-gans-fa847f696e04) |
Reference-Based Sketch Image Colorization using Augmented-Self Reference and Dense Semantic Correspondence(CVPR 2020)[[pdf]](https://arxiv.org/abs/2005.05207) | <ul><li> Augmented-Self Reference Generation </il><li> Spatially Corresponding Feature Transfer module(for Attribute transfer) | Not implemented(2021/02/14) |[Yes](https://medium.com/analytics-vidhya/paper-review-reference-based-sketch-image-colorization-using-augmented-self-reference-and-dense-4e646f811ff2)
DeshuffleGAN: A Self-Supervised GAN to Improve Structure Learning(IEEE 2020)[[pdf]](https://arxiv.org/abs/2006.08694) | <ul><li> Apply idea that solve Jigsaw puzzle to GAN | Not implemented(2021/02/14) | [Yes](https://medium.com/analytics-vidhya/paper-review-deshufflegan-a-self-supervised-gan-to-improve-structure-learning-1d601f3d95f8) |

### ViewPoint-Estimation
Title | Contributions | Code | review |
--- | --- | --- | --- |
Cylindrical Convolutional Networks for Joint Object Detection and Viewpoint Estimation(CVPR 2020)[[pdf]](https://arxiv.org/abs/2003.11303) | <ul><li> Extract the view-specific feature conditioned on the object viewpoint that encodes structural information at each viewpoint <il><li> Differentiable argmax operator called sinusoidal soft-argmax that can manage sinusoidal | [code](https://github.com/sunghunjoung/CCNs/) | [Yes](https://medium.com/@yw_nam/paper-review-cylindrical-convolutional-networks-for-joint-object-detection-and-viewpoint-813acead4b2c)

### Image-Synthesis,Image-Transfer
Title | Contributions | Code | review |
--- | --- | --- | --- |
ADGAN:Controllable Person Image Synthesis With Attribute-Decomposed GAN(CVPR 2020)[[pdf]](https://arxiv.org/pdf/2003.12267.pdf) | <ul><li> Cloth Transfer by Decomposed Component Encoder and Global Texture Encoder <il><li> Texture Style Transfer using Fusion module | [code](https://github.com/menyifang/ADGAN) | [Yes](https://medium.com/analytics-vidhya/paper-review-adgan-controllable-person-image-synthesis-with-attribute-decomposed-gan-1c45bddbe00a) |
Spatially Controllable Image Synthesis with Internal Representation Collaging[[pdf]](https://arxiv.org/pdf/1811.10153.pdf) | <ul><li> Spatial Conditional batch normalization(called sCBN), conditinoal batch norm with user-specifiable spatial weight map <il><li> Feature blending by directly modifying the intermediate features | [code](https://github.com/quolc/neural-collage) | - |
  
### Image-Augmentation
Title | Contributions | Code | review |
--- | --- | --- | --- |
Robust and Generalizable Visual Representation Learning via Random Convolutions[[pdf]](https://arxiv.org/pdf/1811.10153.pdf) | <ul><li> explore using outputs of multi-scale random convolutions as new images or mixing them with the original images during training | Not implented yet(2021/02/15) | - |

### Dataset
Title | Contributions | Code | review |
--- | --- | --- | --- |
DanbooRegion: An Illustration Region Dataset(ECCV 2020)[[pdf]](https://lllyasviel.github.io/DanbooRegion/paper/paper.pdf) | <ul><li> Dataset that annotate regions for in-the-wild cartoon images | [[code]](https://github.com/lllyasviel/DanbooRegion) | - |
  

## Natural-Language-Processing
### Audio-Synthesis
Title | Contributions | Code | review |
--- | --- | --- | --- |
WAVENET: A GENERATIVE MODEL FOR RAW AUDIO [[pdf]](https://arxiv.org/pdf/1609.03499.pdf) | <ul><li> develop new architectures based on dilated causal convolutions, which exhibit very large receptive fields <ul><li> Show that when conditioned on a speaker identity, a single model can be used to generate different voices. large receptive fields. | [[code]](https://github.com/ibab/tensorflow-wavenet) | - | 


## Data-Privacy

Title | Contributions | Code | review |
--- | --- | --- | --- |
Learn From Distributed Asynchronized GAN Without Sharing Medical Image Data(CVPR 2020)[[pdf]](https://arxiv.org/abs/2006.00080) | <ul><li> In clinical environment, Prviacy violation is critical point. So, this paper handle this problem by Synthesis image using GAN | [code](https://github.com/tommy-qichang/AsynDGAN) | [Yes](https://medium.com/analytics-vidhya/paper-review-asyndgan-train-deep-learning-without-sharing-medical-image-data-ac93b5592be4) |
Unsupervised Model Personalization while Preserving Privacy and Scalability An Open Problem(CVPR 2020)[[pdf]](https://arxiv.org/abs/2003.13296) | <ul><li> DUA framework-a single model that multiple task-specific model compressed | [code](https://github.com/mattdl/DUA) | - | 
  
## Medical-application
Title | site |
--- | --- |
Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs | [[site]](https://pubmed.ncbi.nlm.nih.gov/27898976/) |
Fundus Image Classification Using VGG-19 Architecture with PCA and SVD | [[site]](https://www.mdpi.com/2073-8994/11/1/1) |
