# Paper List
The list of paper that i read

## Contents
- [Computer-Vision](#Computer-Vision)
  - [Self-Supervised-Learning](#Self-Supervised-Learning)
  - [Image-Synthesis,Image-Transfer](#Image-Synthesis,Image-Transfer)
  - [Dataset](#Dataset) 
  - [Others](#Others)
- [Natural-Language-Processing](#Natural-Language-Processing)
  - [Audio-Synthesis](#Audio-Synthesis)
- [Others](#Others)
- [Medical-application](#Medical-application)

## Computer-Vision
### Self-Supervised Learning

Title | Contributions | Code | review |
--- | --- | --- | --- |
Diverse Image Generation via Self-Conditioned GANs(CVPR 2020)[[pdf]](https://arxiv.org/abs/2006.10728) | <ul><li> Self-Conditional GAN by clustering </il><li> Computing Dataset partition by clustering | [code](https://github.com/stevliu/self-conditioned-gan) | [Yes](https://medium.com/analytics-vidhya/paper-review-diverse-image-generation-via-self-conditioned-gans-fa847f696e04) |
Reference-Based Sketch Image Colorization using Augmented-Self Reference and Dense Semantic Correspondence(CVPR 2020)[[pdf]](https://arxiv.org/abs/2005.05207) | <ul><li> Augmented-Self Reference Generation </il><li> Spatially Corresponding Feature Transfer module(for Attribute transfer) | Not implemented(2021/02/14) |[Yes](https://medium.com/analytics-vidhya/paper-review-reference-based-sketch-image-colorization-using-augmented-self-reference-and-dense-4e646f811ff2)
DeshuffleGAN: A Self-Supervised GAN to Improve Structure Learning(IEEE 2020)[[pdf]](https://arxiv.org/abs/2006.08694) | <ul><li> Apply idea that solve Jigsaw puzzle to GAN | Not implemented(2021/02/14) | [Yes](https://medium.com/analytics-vidhya/paper-review-deshufflegan-a-self-supervised-gan-to-improve-structure-learning-1d601f3d95f8) |
Self-Supervised GANs via Auxiliary Rotation Loss(CVPR 2018) [[pdf]](https://arxiv.org/abs/1811.11212) | <ui><li>Unsupervised generative model that combines adversarial training with self-supervised learning | [[code]](https://github.com/google/compare_gan) | - |
Steering Self-Supervised Feature Learning Beyond Local Pixel Statistics(CVPR 2020) [[pdf]](https://arxiv.org/abs/2004.02331) | <ul><li> Introduce a novel selfsupervised learning principle based on image transformations that can be detected only through global observations  </il><li> Introduce a novel transformation according to this principle and demonstrate experimentally its impact on feature learning <li><il> Formulate the method so that it can easily scale with additional transformations | [[code]](https://github.com/sjenni/LCI) | - |
Self-PU: Self Boosted and Calibrated Positive-Unlabeled Training(ICML 2020) [[pdf]](https://arxiv.org/abs/2006.11280) | <ul><li> A self-distillation scheme is designed via the collaborative training between several teacher networks and student networks </il><li> A self-calibration strategy is leveraged to further explore the fine-grained treatment of loss functions over unconfident examples, in a meta-learning fashion | [[code]](https://github.com/VITA-Group/Self-PU) | - |
Unsupervised Part Discovery via Feature Alignment(submitted CVPR 2021 - 2021/02/15) [[pdf]](https://arxiv.org/abs/2012.00313) | <ul><li> Part detectors that are consistent across different object instances, 3D poses and articulations  </il><li> The authors argue that Without groundtruth part annotation, there still exists supervisions, coming from the coherence between similar images | Not implemented yet(2021/02/15) | - |


### Image-Synthesis,Image-Transfer
Title | Contributions | Code | review |
--- | --- | --- | --- |
ADGAN:Controllable Person Image Synthesis With Attribute-Decomposed GAN(CVPR 2020)[[pdf]](https://arxiv.org/pdf/2003.12267.pdf) | <ul><li> Cloth Transfer by Decomposed Component Encoder and Global Texture Encoder <il><li> Texture Style Transfer using Fusion module | [code](https://github.com/menyifang/ADGAN) | [Yes](https://medium.com/analytics-vidhya/paper-review-adgan-controllable-person-image-synthesis-with-attribute-decomposed-gan-1c45bddbe00a) |
Spatially Controllable Image Synthesis with Internal Representation Collaging[[pdf]](https://arxiv.org/pdf/1811.10153.pdf) | <ul><li> Spatial Conditional batch normalization(called sCBN), conditinoal batch norm with user-specifiable spatial weight map <il><li> Feature blending by directly modifying the intermediate features | [code](https://github.com/quolc/neural-collage) | - |

### Dataset
Title | Contributions | Code | review |
--- | --- | --- | --- |
DanbooRegion: An Illustration Region Dataset(ECCV 2020)[[pdf]](https://lllyasviel.github.io/DanbooRegion/paper/paper.pdf) | <ul><li> Dataset that annotate regions for in-the-wild cartoon images | [[code]](https://github.com/lllyasviel/DanbooRegion) | - |
 
### Others
Title | Contributions | Code | review |
--- | --- | --- | --- |
Robust and Generalizable Visual Representation Learning via Random Convolutions[[pdf]](https://arxiv.org/pdf/1811.10153.pdf) | <ul><li> Explore using outputs of multi-scale random convolutions as new images or mixing them with the original images during training | Not implented yet(2021/02/15) | - |
Spectral Normalization for Generative Adversarial Networks(ICLR 2018) [[pdf]](https://arxiv.org/pdf/1802.05957.pdf) | <ul><li> Lipschitz constant is the only hyper-parameter to be tuned, and the algorithm does not require intensive tuning of the only hyper-parameter for satisfactory performance. <il><li> Functioned well even without tuning Lipschitz constant, which is the only hyper parameter | [[code]](https://github.com/pfnet-research/sngan_projection) | - |


## Natural-Language-Processing
### Audio-Synthesis
Title | Contributions | Code | review |
--- | --- | --- | --- |
WAVENET: A GENERATIVE MODEL FOR RAW AUDIO [[pdf]](https://arxiv.org/pdf/1609.03499.pdf) | <ul><li> Develop new architectures based on dilated causal convolutions, which exhibit very large receptive fields <il><li> Show that when conditioned on a speaker identity, a single model can be used to generate different voices. large receptive fields. | [[code]](https://github.com/ibab/tensorflow-wavenet) | - | 


## Others
Title | Contributions | Code | review |
--- | --- | --- | --- |
Learn From Distributed Asynchronized GAN Without Sharing Medical Image Data(CVPR 2020)[[pdf]](https://arxiv.org/abs/2006.00080) | <ul><li> In clinical environment, Prviacy violation is critical point. So, this paper handle this problem by Synthesis image using GAN | [code](https://github.com/tommy-qichang/AsynDGAN) | [Yes](https://medium.com/analytics-vidhya/paper-review-asyndgan-train-deep-learning-without-sharing-medical-image-data-ac93b5592be4) |
Unsupervised Model Personalization while Preserving Privacy and Scalability An Open Problem(CVPR 2020)[[pdf]](https://arxiv.org/abs/2003.13296) | <ul><li> DUA framework-a single model that multiple task-specific model compressed | [code](https://github.com/mattdl/DUA) | - | 
Self-Supervised Learning for Generalizable Out-of-Distribution Detection(AAAI 2020)[[pdf]](http://people.tamu.edu/~sina.mohseni/papers/Self_Supervised_Learning_for_Generalizable_Out_of_Distribution_Detection.pdf) | <ul><li> The Techique does not need to pre-know the distribution of targeted OOD samples and incur no extra overhead | Not implemented yet(2021/02/15) | - | 
Cylindrical Convolutional Networks for Joint Object Detection and Viewpoint Estimation(CVPR 2020)[[pdf]](https://arxiv.org/abs/2003.11303) | <ul><li> Extract the view-specific feature conditioned on the object viewpoint that encodes structural information at each viewpoint <il><li> Differentiable argmax operator called sinusoidal soft-argmax that can manage sinusoidal | [code](https://github.com/sunghunjoung/CCNs/) | [Yes](https://medium.com/@yw_nam/paper-review-cylindrical-convolutional-networks-for-joint-object-detection-and-viewpoint-813acead4b2c)
  
  
## Medical-application
Title | site |
--- | --- |
Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs | [[site]](https://pubmed.ncbi.nlm.nih.gov/27898976/) |
Fundus Image Classification Using VGG-19 Architecture with PCA and SVD | [[site]](https://www.mdpi.com/2073-8994/11/1/1) |
Artificial Intelligence and Its Effect on Dermatologists’ Accuracy in Dermoscopic Melanoma Image Classification: Web-Based Survey Study | [[site]](https://www.jmir.org/2020/9/e18091/) |
Joint Optic Disc and Cup Segmentation Based on Multi-label Deep Network and Polar Transformation | [[site]](https://arxiv.org/abs/1801.00926)
