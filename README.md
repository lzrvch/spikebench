## pyspikelib: A set of tools for neuronal spiking data mining

By [Ivan Lazarevich](https://lazarevi.ch)

**pyspikelib** allows building machine learning models on time series representations of neuronal spike trains. For more details, please refer to our [arXiv paper](https://arxiv.org/abs/1810.03855).

<p align="center">
    <img src="https://i.imgur.com/7xqACVK.jpg" alt="drawing" width="500"/>
</p>

# Citing pyspikelib
Please cite this work in your publications if it helps your research:

```
@article{lazarevich2020neural,
  title={Neural activity classification with machine learning models trained on interspike interval series data},
  author={Lazarevich, Ivan and Prokin, Ilya and Gutkin, Boris},
  journal={arXiv preprint arXiv:1810.03855},
  year={2020}
}
```

# Usage
Install the package by running
```
make install
```

After downloading the retinal neuronal activity dataset (see details below; similarly for other datasets) one can run the scripts in the examples folder e.g.

```
python examples/retina_example.py --seed 0 --feature-set "no_entropy"
```

# Dataset sources

  - fcx-1 dataset is taken from the [CRCNS database](http://crcns.org/data-sets/fcx/fcx-1/about-fcx-1). The processed CRNCS fcx-1 data from the example can be downloaded [via this link](https://drive.google.com/open?id=1fQKpYPHmenob692YZaG1P7YKWCYaTw19) (filesize is approximately 26 Mb).
  - example of retinal ganglion cell activity classification is based on [the published dataset](https://figshare.com/articles/Multi-electrode_retinal_ganglion_cell_population_spiking_data/10290569)
  - the interneuron subtype classification example is based on data fetched from the [Allen Cell Types dataset](https://celltypes.brain-map.org/).
