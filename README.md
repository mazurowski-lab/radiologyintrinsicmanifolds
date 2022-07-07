# Code for MICCAI 2022 paper "The Intrinsic Manifolds of Radiological Images and their Role in Deep Learning"
## by Nicholas Konz, Hanxue Gu, Haoyu Dong and Maciej Mazurowski

![Example images from our explored datasets.](figures/data_eg_1row.png)

### Paper Abstract
The manifold hypothesis is a core mechanism behind the success of deep learning, so understanding the intrinsic manifold structure of image data is central to studying how neural networks learn from the data. Intrinsic dataset manifolds and their relationship to learning difficulty have recently begun to be studied for the common domain of natural images, but little such research has been attempted for radiological images. We address this here. First, we compare the intrinsic manifold dimensionality of radiological and natural images. We also investigate the relationship between intrinsic dimensionality and generalization ability over a wide range of datasets. Our analysis shows that natural image datasets generally have a higher number of intrinsic dimensions than radiological images. However, the relationship between generalization ability and intrinsic dimensionality is much stronger for medical images, which could be explained as radiological images having intrinsic features that are more difficult to learn. These results give a more principled underpinning for the intuition that radiological images can be more challenging to apply deep learning to than natural image datasets common to machine learning research.  We believe rather than directly applying models developed for natural images to the radiological imaging domain, more care should be taken to developing architectures and algorithms that are more tailored to the specific characteristics of this domain. The research shown in our paper, demonstrating these characteristics and the differences from natural images, is an important first step in this direction.

See the paper (preprint version) here: https://arxiv.org/abs/2207.02797.

## Main Findings

1. Natural image datasets generally have higher intrinsic dimension than radiology datasets:
![Intrinsic dimension of various radiological and natural image datasets.](figures/ID.png)
3. The relationship of test prediction accuracy and dataset intrinsic dimension is linear *within* both of these two domains, but,
4. The *steepness* of this relationship greatly differs between natural and radiological domains:
![Difference in generalization ability vs. dataset intrinsic dimension between natural and radiological images.](figures/main_fig_multi_0.png)

A future endeavor is to determine the theoretical reasons for all of these findings.

## Reproducing the Results
Please follow the steps outlined in `reproducibility_tutorial.md` in order to reproduce the results of the paper.

## Citation
*forthcoming*
