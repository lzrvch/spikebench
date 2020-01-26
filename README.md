




# PySpikeLib: A set of tools for neuronal spiking data mining

By [Ivan Lazarevich](https://lazarevi.ch)

# Introduction

PySpikeLib allows building machine learning models on time series representations of neuronal spike trains. For more details, please refer to our [arXiv paper](https://arxiv.org/abs/1810.03855).

To get started, have a look at an example in the `notebooks` subfolder which is focused on predicting sleep vs. quiet wakefulness states of an animal from the spike trains of individual neurons in the frontal cortex. The data for this example is taken from the [fcx-1 dataset in the CRCNS repository](http://crcns.org/data-sets/fcx/fcx-1/about-fcx-1). This example of building a classifier to predict an animal state is covered in-depth in [our paper](https://arxiv.org/abs/1810.03855)

<img src="https://i.imgur.com/7xqACVK.jpg" alt="drawing" width="500"/>

# Citing PySpikeLib
Please cite this wotk in your publications if it helps your research:

```
@article{lazarevich2020neural,
  title={Neural activity classification with machine learning models trained on interspike interval series data},
  author={Lazarevich, Ivan and Prokin, Ilya and Gutkin, Boris},
  journal={arXiv preprint arXiv:1810.03855},
  year={2020}
}
```

# Getting CRCNS.org fcx-1 data to run WAKE/SLEEP classification example 

The processed CRNCS fcx-1 data from the example can be downloaded [via this link](https://drive.google.com/open?id=1fQKpYPHmenob692YZaG1P7YKWCYaTw19) (filesize is approximately 26 Mb).
