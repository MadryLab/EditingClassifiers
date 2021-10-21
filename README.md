This repository contains the code and data for our paper:

**Editing a classifier by rewriting its prediction rules** <br>
*Shibani Santurkar\*, Dimitris Tsipras\*, Mahi Elango, David Bau, Antonio Torralba, Aleksander Madry* <br>
Paper: https://arxiv.org/abs/2112.01008 <br>

![](edit_examples.png)

```bibtex
    @InProceedings{santurkar2021editing,
        title={Editing a classifier by rewriting its prediction rules},
        author={Shibani Santurkar and Dimitris Tsipras and Mahalaxmi Elango and David Bau and Antonio Torralba and Aleksander Madry},
        year={2021},
        booktitle={Neural Information Processing Systems (NeurIPS)}
    }
```

## Getting started
You can start by cloning our repository and following the steps below. Parts of this codebase have been derived from the [GAN rewriting
repository](https://github.com/davidbau/rewriting) of Bau et al. 

1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yaml
    ```

2. Download [model checkpoints](https://github.com/MadryLab/EditingClassifiers/releases/download/v1/checkpoints.tar.gz) and extract them in the current directory.

3. To instantiate CLIP
    ```
    git submodule init
    git submodule update

    ```
    
4. Replace `IMAGENET_PATH` in `helpers/classifier_helpers.py` with the path to the ImageNet dataset.
 
5. (If using synthetic examples) Download files [segmentations.tar.gz](https://github.com/MadryLab/EditingClassifiers/releases/download/v1/segmentations.tar.gz) and [styles.tar.gz](https://github.com/MadryLab/EditingClassifiers/releases/download/v1/styles.tar.gz) and extract them under `./data/synthetic`.

6. (If using synthetic examples) Run 
```
     python stylize.py --style_name [STYLE_FILE_NAME]
```
with the desired style file from `./data/synthetic/styles`. You could also use a custom style file if desired.

That's it! Now you can explore our editing methodology in various settings: [vehicles-on-snow](https://github.com/MadryLab/EditingClassifiers/blob/master/vehicles_on_snow.ipynb), [typographic attacks](https://github.com/MadryLab/EditingClassifiers/blob/master/typographic_attacks.ipynb) and [synthetic test cases](https://github.com/MadryLab/EditingClassifiers/blob/master/synthetic_test_cases.ipynb).

# Maintainers

* [Shibani Santurkar](https://twitter.com/ShibaniSan)
* [Dimitris Tsipras](https://twitter.com/tsiprasd)
