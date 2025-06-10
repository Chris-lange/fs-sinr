# Feedforward Few-shot Species Range Estimation - ICML 2025

Code for training and evaluating global-scale species range estimation models. This code enables the recreation of the results from our ICML 2025 paper [Feedforward Few-shot Species Range Estimation](https://arxiv.org/abs/2502.14977). 

## 🌍 Overview 
Knowing where a particular species can or cannot be found on Earth is crucial for ecological research and conservation efforts. By mapping the spatial ranges of all species, we would obtain deeper insights into how global biodiversity is affected by climate change and habitat loss. However, accurate range estimates are only available for a relatively small proportion of all known species. For the majority of the remaining species, we typically only have a small number of records denoting the spatial locations where they have previously been observed. We outline a new approach for few-shot species range estimation to address the challenge of accurately estimating the range of a species from limited data. During inference, our model takes a set of spatial locations as input, along with optional metadata such as text or an image, and outputs a species encoding that can be used to predict the range of a previously unseen species in a feedforward manner. We evaluate our approach on two challenging benchmarks, where we obtain state-of-the-art range estimation performance, in a fraction of the compute time, compared to recent alternative approaches. 

## 🔍 Getting Started 

#### Installing Required Packages

1. We recommend using an isolated Python environment to avoid dependency issues.

2. Create a new environment and activate it:
```bash
 conda create -y --name fs_sinr_icml python==3.11
 conda activate fs_sinr_icml
```

3. After activating the environment, install the required packages:
```bash
 pip3 install -r requirements.txt
```

#### Data Download and Preparation

Instructions for downloading the data in `data/README.md`.

## 🗺️ Generating Predictions

To generate predictions for an evaluation species in the form of an image, run the following command: 
```bash
 python viz_fs_species_range.py
```
Use an appropriate inaturalist id number for a species in the S&T or IUCN evaluation datasets, such as the [Yellow-footed Green Pigeon](https://www.inaturalist.org/taxa/3352).

To generate similar predictions using a text prompt and / or an image, run the following command: 
```bash
 python viz_fs_map_non_species.py
```
Process images that you wish to use for this with `image_to_representation.py`

## 🚅 Training and Evaluating Models

To train a model, run the following command:
```bash
 python train_model.py
```
To evaluate a trained model, run the following command:
```bash
 python evaluate_model.py
```

#### Hyperparameters
Common parameters of interest can be set as arguments in `train_model.py` and `evaluate_model.py`. All other parameters are exposed in `setup.py`. 

#### Outputs
By default, trained models and evaluation results will be saved to a folder in the `experiments` directory. Evaluation results will also be printed to the command line. 

##  🙏 Acknowledgements
We thank the iNaturalist community for making the species observation data available. Oisin Mac Aodha was supported by a Royal Society Research Grant. Max Hamilton and Subhransu Maji were supported by NSF grants 2329927 and 2406687.

If you find our work useful in your research please consider citing our paper.  
```
@inproceedings{lange2025fewshot,
  title     = {Feedforward Few-shot Species Range Estimation},
  author    = {Lange, Christian and Hamilton, Max and Cole, Elijah and Shepard, Alexander and Heinrich, Samuel and Zhu, Angela and Maji, Subhransu and Van Horn, Grant and Mac Aodha, Oisin},
  booktitle = {The Proceedings of the 42nd International Conference on Machine Learning},
  year = {2025}
}
```

## 📜 Disclaimer
Our models rely on text embeddings and summaries generated from LLMs, and thus may inherit the biases contained within them. Both Wikipedia text and observational data from iNaturalist are biased toward the United States and Western Europe. As a result, there could be potential negative consequences associated with using the species range predictions from our model to inform conservation or policy decisions. Therefore, caution is encouraged when making decisions based on the model’s predictions.
Extreme care should be taken before making any decisions based on the outputs of models presented here. Our goal in this work is to demonstrate the promise of large-scale representation learning for few-shot species range estimation, not to provide definitive range maps. Our models are trained on biased data and have not been calibrated or validated beyond the experiments illustrated in the paper. 

