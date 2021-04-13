# Paper List
The list of paper that i read

## Contents
- [Computer Vision](#Computer-Vision)
  - [Self Supervised Learning](#Self-Supervised-Learning)
  - [Image Synthesis](#Image-Synthesis)
- [Generative Adversarial Network](#Generative-Adversarial-Network)
- [Dataset](#Dataset) 
- [Others](#Others)
- [Medical Application](#Medical-application)

## Computer-Vision
### Self-Supervised Learning

Title | Contributions | Code | Review | Recommand|
--- | --- | --- | --- | --- | 
Diverse Image Generation via Self-Conditioned GANs(CVPR 2020)[[pdf]](https://arxiv.org/abs/2006.10728) | <ul><li> Self-Conditional GAN by clustering <il><li> Computing Dataset partition by clustering | [code](https://github.com/stevliu/self-conditioned-gan) | [Yes](https://medium.com/analytics-vidhya/paper-review-diverse-image-generation-via-self-conditioned-gans-fa847f696e04) | :star::star::star: |
Reference-Based Sketch Image Colorization using Augmented-Self Reference and Dense Semantic Correspondence(CVPR 2020)[[pdf]](https://arxiv.org/abs/2005.05207) | <ul><li> Augmented-Self Reference Generation </il><li> Spatially Corresponding Feature Transfer module(for Attribute transfer) | Not implemented(2021/02/14) |[Yes](https://medium.com/analytics-vidhya/paper-review-reference-based-sketch-image-colorization-using-augmented-self-reference-and-dense-4e646f811ff2) | :star::star::star::star::star:|
DeshuffleGAN: A Self-Supervised GAN to Improve Structure Learning(IEEE 2020)[[pdf]](https://arxiv.org/abs/2006.08694) | <ul><li> Apply idea that solve Jigsaw puzzle to GAN | Not implemented(2021/02/14) | [Yes](https://medium.com/analytics-vidhya/paper-review-deshufflegan-a-self-supervised-gan-to-improve-structure-learning-1d601f3d95f8) | :star::star::star::star: |
Self-Supervised GANs via Auxiliary Rotation Loss(CVPR 2018) [[pdf]](https://arxiv.org/abs/1811.11212) | <ul><li>Unsupervised generative model that combines adversarial training with self-supervised learning | [code](https://github.com/google/compare_gan) | - | :star::star::star::star: | 
Steering Self-Supervised Feature Learning Beyond Local Pixel Statistics(CVPR 2020) [[pdf]](https://arxiv.org/abs/2004.02331) | <ul><li> Introduce a novel selfsupervised learning principle based on image transformations that can be detected only through global observations  </il><li> Introduce a novel transformation according to this principle and demonstrate experimentally its impact on feature learning <li><il> Formulate the method so that it can easily scale with additional transformations | [code](https://github.com/sjenni/LCI) | - | :star::star::star: |
Self-PU: Self Boosted and Calibrated Positive-Unlabeled Training(ICML 2020) [[pdf]](https://arxiv.org/abs/2006.11280) | <ul><li> A self-distillation scheme is designed via the collaborative training between several teacher networks and student networks </il><li> A self-calibration strategy is leveraged to further explore the fine-grained treatment of loss functions over unconfident examples, in a meta-learning fashion | [code](https://github.com/VITA-Group/Self-PU) | - | :star::star::star: |
Unsupervised Part Discovery via Feature Alignment(submitted CVPR 2021 - 2021/02/15) [[pdf]](https://arxiv.org/abs/2012.00313) | <ul><li> Part detectors that are consistent across different object instances, 3D poses and articulations  </il><li> The authors argue that Without groundtruth part annotation, there still exists supervisions, coming from the coherence between similar images | Not implemented yet(2021/02/15) | - | :star::star::star::star: |
Self-Supervised Learning for Generalizable Out-of-Distribution Detection(AAAI 2020)[[pdf]](http://people.tamu.edu/~sina.mohseni/papers/Self_Supervised_Learning_for_Generalizable_Out_of_Distribution_Detection.pdf) | <ul><li> The Techique does not need to pre-know the distribution of targeted OOD samples and no extra overhead | Not implemented yet(2021/02/15) | - | :star::star::star: | 
SCOPS: Self-Supervised Co-Part Segmentation(CVPR 2019)[[pdf]](https://arxiv.org/abs/1905.01298) | <ul><li> Propose a self-supervised deep learning framework for part segmentation that use given an image collection of the same object category. <il><li> Devise loss functions for part segmentations | [code](https://github.com/NVlabs/SCOPS) | - | :star::star::star::star: |

  
### Image-Synthesis
Title | Contributions | Code | Review | Recommand |
--- | --- | --- | --- | --- |
Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks(2017 ICCV)[[pdf]](https://arxiv.org/abs/1703.10593) | <ul><li> Propose method that capturing special characteristics of one image collection and figuring out how these characteristics could be translated into the other image collection <il><li> Propose algorithm that can learn to translate between domains without paired input-output | [code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) | - | :star::star::star::star::star: |
A Robust Pose Transformational GAN for Pose Guided Person Image Synthesis(NCVPRIPG 2019)[[pdf]](https://arxiv.org/abs/2001.01259) | <ul><li> Aim to develop an improved end-to-end model for pose transformation given only the input image and the desired pose, and without any other external features <il><li> Achieve robustness in terms of occlusion, scale and illumination by efficient data augmentation techniques and utilizing inherent CNN features | Not implemented yet(2021/02/16) | - | :star::star: |
ADGAN:Controllable Person Image Synthesis With Attribute-Decomposed GAN(CVPR 2020)[[pdf]](https://arxiv.org/pdf/2003.12267.pdf) | <ul><li> Cloth Transfer by Decomposed Component Encoder and Global Texture Encoder <il><li> Texture Style Transfer using Fusion module | [code](https://github.com/menyifang/ADGAN) | [Yes](https://medium.com/analytics-vidhya/paper-review-adgan-controllable-person-image-synthesis-with-attribute-decomposed-gan-1c45bddbe00a) | :star::star::star::star::star: |
Spatially Controllable Image Synthesis with Internal Representation Collaging[[pdf]](https://arxiv.org/pdf/1811.10153.pdf) | <ul><li> Spatial Conditional batch normalization(called sCBN), conditinoal batch norm with user-specifiable spatial weight map <il><li> Feature blending by directly modifying the intermediate features | [code](https://github.com/quolc/neural-collage) | - | :star::star::star: |
Soft-Gated Warping-GAN for Pose-Guided Person Image Synthesis [[pdf]](https://arxiv.org/abs/1810.11610) | <ul><li> Soft-Gated Warping-GAN to address the large spatial misalignment issues induced by geometric transformations of desired poses | Not implemented yet(2021/02/15) | - | :star::star::star: |
Deep Image Spatial Transformation for Person Image Generation(CVPR 2020) [[pdf]](https://arxiv.org/abs/2003.00696) | <ul><li> Propose a differentiable global-flow local-attention framework to reassemble the inputs at the feature level <il><li> video animation and view synthesis show that our model is applicable to other tasks requiring spatial transformation | [code](https://github.com/RenYurui/Global-Flow-Local-Attention) | - | :star::star::star::star: |
StarGAN v2: Diverse Image Synthesis for Multiple Domains(CVPR 2020)[[pdf]](https://arxiv.org/abs/1912.01865) | <ul><li> The mapping network learns to transform random Gaussian noise into a style code <il><li> Style encoder learns to extract the style code from a given reference image <il><li> Utilizing these style codes, our generator learns to successfully synthesize diverse images over multiple domains | [code](https://github.com/clovaai/stargan-v2) | - | :star::star::star::star::star: |
Disentangling Style and Content in Anime Illustrations[[pdf]](https://arxiv.org/abs/1905.10742) | <ul><li> Demonstrate the ability to generate high-fidelity anime portraits with a fixed content and a large variety of styles from over a thousand artists | [code](https://github.com/stormraiser/adversarial-disentangle) | - | :star::star::star: |
Style Transfer for Anime Sketches with Enhanced Residual U-net and Auxiliary Classifier GAN(ACPR 2017)[[pdf]](https://arxiv.org/abs/1706.03319) | <ul><li> A feed-forward network to apply the style of a painting to a sketch. <il><li> An enhanced residual U-net capable of handling paired images with unbalanced information quantity.<il><li> An effective way to train residual U-net with two additional loss. <il><li>A discriminator modified from AC-GAN suitable to deal with paintings of different style. | [code](https://paperswithcode.com/paper/style-transfer-for-anime-sketches-with) | - | :star::star::star: |
Tag2Pix: Line Art Colorization Using Text Tag With SECat and Changing Loss(ICCV 2019)[[pdf]](https://arxiv.org/abs/1908.05840) | <ul><li> Propose a dataset for training the Tag2Pix network <il><li> Present an adversarial line art colorization network called Tag2Pix <il><li> propose a novel network structure called SE-cat that enhances multi-label segmentation and colorization. <il><li> Present a novel loss combination and curriculum learning method for the Tag2Pix network | [code](https://github.com/blandocs/Tag2Pix) | - |  :star::star::star::star: |

## Generative Adversarial Network
Title | Contributions | Code | Review | Recommand |
--- | --- | --- | --- | --- |
Generative Adversarial Networks(2014 NIPS)[[pdf]](https://arxiv.org/abs/1406.2661) | <ul><li> Propose adversarial network | [code](https://github.com/eriklindernoren/PyTorch-GAN) | - | :star::star::star::star::star: |
Conditional Generative Adversarial Nets[[pdf]](https://arxiv.org/abs/1411.1784) | <ul><li> Propose method that give condition to gan | [code](https://github.com/eriklindernoren/PyTorch-GAN) | - | :star::star::star::star::star: |
On Convergence and Stability of GANs[[pdf]](https://arxiv.org/abs/1705.07215) | <ul><li> Propose a new way of reasoning about the GAN training dynamics - by viewing AGD as regret minimization. <il><li> Provide a novel proof for the asymptotic convergence of GAN training in the nonparametric limit and it does not require the discriminator to be optimal at each step <il><li> Propose Gradient penalty scheme called DRAGAN | [code](https://github.com/kodalinaveen3/DRAGAN) | - | :star::star::star: |
Spectral Normalization for Generative Adversarial Networks(ICLR 2018) [[pdf]](https://arxiv.org/pdf/1802.05957.pdf) | <ul><li> Lipschitz constant is the only hyper-parameter to be tuned, and the algorithm does not require intensive tuning of the only hyper-parameter for satisfactory performance. <il><li> Functioned well even without tuning Lipschitz constant, which is the only hyper parameter | [code](https://github.com/pfnet-research/sngan_projection) | - | :star::star::star::star::star: |
Learn From Distributed Asynchronized GAN Without Sharing Medical Image Data(CVPR 2020)[[pdf]](https://arxiv.org/abs/2006.00080) | <ul><li> In clinical environment, Prviacy violation is critical point. So, this paper handle this problem by Synthesis image using GAN | [code](https://github.com/tommy-qichang/AsynDGAN) | [Yes](https://medium.com/analytics-vidhya/paper-review-asyndgan-train-deep-learning-without-sharing-medical-image-data-ac93b5592be4) | :star::star::star: |
Data Synthesis based on Generative Adversarial Networks [[pdf]](https://arxiv.org/abs/1806.03384) | <ul><li> Propose Table-GAN, uses generative adversarial networks (GANs) to synthesize fake tables that are statistically similar to the original table yet do not incur information leakage. | [code](https://github.com/mahmoodm2/tableGAN) | - | :star::star: |
PerceptionGAN: Real-world Image Construction from Provided Text through Perceptual Understanding(IVPR 2020) [[pdf]](https://arxiv.org/abs/2007.00977) | <ul><li> Propose a method to address the first problem, namely, generating a good, perceptually relevant, low-resolution image to be used as an initialization for the refinement stage | Not implemented yet(2021/02/15) | - | :star::star::star: |
 
## Dataset
Title | Contributions | Code | Review | Recommand |
--- | --- | --- | --- | --- |
DanbooRegion: An Illustration Region Dataset(ECCV 2020)[[site]](https://lllyasviel.github.io/DanbooRegion) | <ul><li> Dataset that annotate regions for in-the-wild cartoon images | [code](https://github.com/lllyasviel/DanbooRegion) | - | :star::star::star: |

  
## Others
Title | Contributions | Code | review | Recommand |
--- | --- | --- | --- | --- |
Unsupervised Model Personalization while Preserving Privacy and Scalability An Open Problem(CVPR 2020)[[pdf]](https://arxiv.org/abs/2003.13296) | <ul><li> DUA framework-a single model that multiple task-specific model compressed | [code](https://github.com/mattdl/DUA) | - | :star::star::star: |
Cylindrical Convolutional Networks for Joint Object Detection and Viewpoint Estimation(CVPR 2020)[[pdf]](https://arxiv.org/abs/2003.11303) | <ul><li> Extract the view-specific feature conditioned on the object viewpoint that encodes structural information at each viewpoint <il><li> Differentiable argmax operator called sinusoidal soft-argmax that can manage sinusoidal | [code](https://github.com/sunghunjoung/CCNs/) | [Yes](https://medium.com/@yw_nam/paper-review-cylindrical-convolutional-networks-for-joint-object-detection-and-viewpoint-813acead4b2c) | :star::star::star: |
SelectiveNet: A Deep Neural Network with an Integrated Reject Option(2019 ICML)[[pdf]](https://arxiv.org/abs/1901.09192) | <ul><li> A selective loss function that optimizes a specified coverage slice using a variant of the interior point optimization method <il><li> SelectiveNet: A three-headed network for end-to-end learning of selective classification models. <il><li> SelectiveNet for regression: Providing the first alternative to costly methods such as MC-dropout or ensemble techniques. | [code](https://github.com/geifmany/SelectiveNet) | - | :star::star::star: |
Uncertainty-Based Rejection Wrappers for Black-Box Classifiers(2020 IEEE)[[pdf]](https://ieeexplore.ieee.org/document/9097854) | <ul><li> present a method for estimating the uncertainty of a black-box model | Not implemented yet(2021/02/22) | - | :star::star::star: |
Robust and Generalizable Visual Representation Learning via Random Convolutions[[pdf]](https://arxiv.org/pdf/1811.10153.pdf) | <ul><li> Explore using outputs of multi-scale random convolutions as new images or mixing them with the original images during training | Not implented yet(2021/02/15) | - | :star::star::star: |
  
## Medical-application
Title | site |
--- | --- |
Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs | [site](https://pubmed.ncbi.nlm.nih.gov/27898976/) |
Fundus Image Classification Using VGG-19 Architecture with PCA and SVD | [site](https://www.mdpi.com/2073-8994/11/1/1) |
Artificial Intelligence and Its Effect on Dermatologists’ Accuracy in Dermoscopic Melanoma Image Classification: Web-Based Survey Study | [site](https://www.jmir.org/2020/9/e18091/) |
Joint Optic Disc and Cup Segmentation Based on Multi-label Deep Network and Polar Transformation | [site](https://arxiv.org/abs/1801.00926) |
A Human-Centered Evaluation of a Deep Learning System Deployed in Clinics for the Detection of Diabetic Retinopathy | [site](https://dl.acm.org/doi/abs/10.1145/3313831.3376718) |
Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network | [site](https://www.nature.com/articles/s41591-018-0268-3) |
Expert-validated estimation of diagnostic uncertainty for deep neural networks in diabetic retinopathy detection | [site](https://www.researchgate.net/publication/334470511_Expert-validated_estimation_of_diagnostic_uncertainty_for_deep_neural_networks_in_diabetic_retinopathy_detection) |
Deep learning versus human graders for classifying diabetic retinopathy severity in a nationwide screening program | [site](https://www.nature.com/articles/s41746-019-0099-8) |


