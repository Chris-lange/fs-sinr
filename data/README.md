# Instructions for Data Preparation

These instructions are mostly copied from https://github.com/cvl-umass/le-sinr, which uses the same location and text data as FS-SINR.

After following these instructions, the `data` directory should have the following structure:
```
data
├── README.md
├── env
│   ├── bioclim_elevation_scaled.npy
│   └── format_env_feats.py
├── eval
│   ├── wiki_data.pt
│   ├── geo_feature
│   │   ├── ABOVE_GROUND_CARBON.tif
│   │   ├── ELEVATION.tif
│   │   ├── LEAF_AREA_INDEX.tif
│   │   ├── NON_TREE_VEGITATED.tif
│   │   ├── NOT_VEGITATED.tif
│   │   ├── POPULATION_DENSITY.tif
│   │   ├── SNOW_COVER.tif
│   │   ├── SOIL_MOISTURE.tif
│   │   └── TREE_COVER.tif
│   ├── geo_prior
│   │   ├── geo_prior_model_meta.csv
│   │   └── geo_prior_model_preds.npz
│   │   └── taxa_subsets.json
│   ├── iucn
│   │   └── iucn_res_5.json
│   └── snt
│       └── snt_res_5.npy
├── masks
│   ├── LAND_MASK.tif
    ├── ocean_mask.npy
    ├── ocean_mask_hr.npy
│   └── USA_MASK.tif
└── train
    ├── wiki_data.pt
    ├── geo_prior_train.csv
    └── geo_prior_train_meta.json
```

## Training & Evaluation Observation Data

1. Navigate to the repository root directory:
```bash
cd /path/to/sinr/
```

2. Download the data file:
```bash
curl -L https://data.caltech.edu/records/b0wyb-tat89/files/data.zip --output data.zip
```

3. Extract the data and clean up:
```bash
unzip -q data.zip
```

4. Clean up:
```bash
rm data.zip
```

## Training & Evaluation Text Data
We provide pre-computed embeddings of text from wikipedia articles for training and evaluation.

1. Download training data [here](https://drive.google.com/file/d/1xv47k6dqex0z33NDQrVgLUTtz6_vjr6i/view) and place it in `data/train`
2. Download evaluation data [here](https://drive.google.com/file/d/1EaMJwfcpmDzGiPZm3464tYBtmfsrWUg0/view) and place it in `data/eval`

## Environmental Features

1. Navigate to the directory for the environmental features:
```
cd /path/to/sinr/data/env
```

2. Download the data:
```bash
curl -L https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_5m_bio.zip --output wc2.1_5m_bio.zip
curl -L https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_5m_elev.zip --output wc2.1_5m_elev.zip
```

3. Extract the data:
```bash
unzip -q wc2.1_5m_bio.zip
unzip -q wc2.1_5m_elev.zip
```

4. Run the formatting script:
```bash
python format_env_feats.py
```

5. Clean up:
```bash
rm *.zip
rm *.tif
```

## Training & Evaluation Image Data
We also provide pre-computed embeddings of images taken from iNaturalist.org for training and evaluation.

1. Download data [here](https://drive.google.com/file/d/1qb_frR3TWD-o4xexMtpbvwOenBssR238/view) and place it in `data`