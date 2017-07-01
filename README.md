#  Generative Models in NNabla

Collection of generative models in [NNabla](https://github.com/sony/nnabla).

## What's in it?

NNabla & Jupyter Notebook implementation of ...

### [Generative Adversarial Networks](https://github.com/TakuyaNakazato0227/nnabla_generative_models/tree/master/GAN)

1. Original GAN [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/GAN/Original_GAN.ipynb)] [[paper](https://arxiv.org/abs/1406.2661)] (2014.06) 
1. Conditional GAN [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/GAN/Conditional_GAN.ipynb)] [[paper](https://arxiv.org/abs/1411.1784)] (2014.11)
1. Adversarially Learned Inference [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/GAN/Adversarially_Learned_Inference.ipynb)] [[paper](https://arxiv.org/abs/1606.00704)] (2016.06)
1. f-GAN [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/GAN/f_GAN.ipynb)] [[paper](https://arxiv.org/abs/1606.00709)] (2016.06)
1. InfoGAN [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/GAN/Info_GAN.ipynb)] [[paper](https://arxiv.org/abs/1606.03657)] (2016.06)
1. Coupled GAN [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/GAN/Coupled_GAN.ipynb)] [[paper](https://arxiv.org/abs/1606.07536)] (2016.06)
1. Energy Based GAN [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/GAN/Energy_Based_GAN.ipynb)] [[paper](https://arxiv.org/abs/1609.03126)] (2016.09)
1. Auxiliary Classifier GAN [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/GAN/Auxiliary_Classifier_GAN.ipynb)] [[paper](https://arxiv.org/abs/1610.09585)] (2016.10)
1. Least Squares GAN [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/GAN/Least_Squares_GAN.ipynb)] [[paper](https://arxiv.org/abs/1611.04076v2)] (2016.11)
1. Mode Regularized GAN [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/GAN/Mode_Regularized_GAN.ipynb)] [[paper](https://arxiv.org/abs/1612.02136)] (2016.12)
1. Generative Adversarial Parallelization [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/GAN/Generative_Adversarial_Parallelization.ipynb)] [[paper](https://arxiv.org/abs/1612.04021)] (2016.12)
1. Wasserstein GAN [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/GAN/Wasserstein_GAN.ipynb)] [[paper](https://arxiv.org/abs/1701.07875)] (2017.01)
1. Boundary Seeking GAN [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/GAN/Boundary_Seeking_GAN.ipynb)] [[paper](https://arxiv.org/abs/1702.08431)] (2017.02)
1. DiscoGAN [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/GAN/Disco_GAN.ipynb)] [[paper](https://arxiv.org/abs/1703.05192)] (2017.03)
1. Boundary Equilibrium GAN [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/GAN/Boundary_Equilibrium_GAN.ipynb)] [[paper](https://arxiv.org/abs/1703.10717)] (2017.03)
1. DualGAN [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/GAN/Dual_GAN.ipynb)] [[paper](https://arxiv.org/abs/1704.02510)] (2017.04)
1. Margin Adaptation for GAN [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/GAN/MA_GAN.ipynb)] [[paper](https://arxiv.org/abs/1704.03817)] (2017.04)
1. Softmax GAN [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/GAN/Softmax_GAN.ipynb)] [[paper](https://arxiv.org/abs/1704.06191)] (2017.04)

### [Variational AutoEncoders](https://github.com/TakuyaNakazato0227/nnabla_generative_models/tree/master/VAE)

1. Original VAE [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/VAE/Original_Variational_AutoEncoder.ipynb)] [[paper](https://arxiv.org/abs/1312.6114)] (2013.12)
1. Conditional VAE [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/VAE/Conditional_Variational_AutoEncoder.ipynb)] [[paper](https://arxiv.org/abs/1406.5298)] (2014.06)
1. Adversarial Autoencoder [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/VAE/Adversarial_AutoEncoder.ipynb)] [[paper](https://arxiv.org/abs/1511.05644)] (2015.11)
1. Denoising VAE [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/VAE/Denoising_Variatinoal_AutoEncoder.ipynb)] [[paper](https://arxiv.org/abs/1511.06406)] (2015.11)
1. Adversarial Variational Bayes [[notebook](https://github.com/TakuyaNakazato0227/nnabla_generative_models/blob/master/VAE/Adversarial_Variational_Bayes.ipynb)] [[paper](https://arxiv.org/abs/1701.04722)] (2017.01)

## Dependencies

* jupyter == '1.0.0'
* nnabla == '0.9.1rc3' 
* scipy == '0.19.1'
* numpy == '1.13.0'
* tensorflow == '1.2.0' (for MNIST data only)

## TODO

* parameter search
* weight clipping
* [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) (2017.04)

## Reference
This repository is inspired by [wiseodd's awesome implementation in Pytorch & Tensorflow](https://github.com/wiseodd/generative-models)