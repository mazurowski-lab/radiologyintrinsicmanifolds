# Reproducibility tutorial for the results of "The Intrinsic Manifolds of Radiological Images and their Role in Deep Learning"

In this document we detail how to reproduce all results of our paper.

### (0) Requirements

You'll need the Python packages outlined in `requirements.txt`, to run with Python 3, along with an updated version of PyTorch. These can be accomplished with:

```
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r requirements.txt

```

### (1) Download the data
This paper involves training models on seven different radiological image datasets. Here we detail how to set up these datasets for further experiments.

#### BraTS

#### CheXpert

1. Download the CheXpert Large dataset from StanfordML [here](https://stanfordmlgroup.github.io/competitions/chexpert/). You'll need to register in their system and follow their instructions. Once you get an email from them, you can either download the data in browser or from the command line (recommended due to the large size) with `wget --show-progress -O chexpert_large.zip {url}`, where `url` is the url that they supply you. Make sure that it saved into a directory named `CheXpert-v1.0`, i.e. `data/chexpert/CheXpert-v1.0`; this directory should have data folders like `train`, `valid` and label `.csvs` like `train.csv` and `valid.csv`.

2. Move the IPython notebook `data/chexpert/make_subset.ipynb` into `data/chexpert/CheXpert-v1.0`, the data folder, and run the cells in it. This will create a subset of the CheXpert (`.png`) training set (CheXpert is quite large, and we don't need all of the data) in `data/chexpert/CheXpert-v1.0/subset/train`, with labels in the new file `data/chexpert/CheXpert-v1.0/train_subset.csv`. 

3. After this feel free to delete the original CheXpert data files to save space.

#### DBC

1. Download the Duke-Breast-Cancer-MRI dataset from The Cancer Imaging Archive [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903). This will take a couple of steps; all files are found under "Data Access" near the bottom of the page.

2. First, download the annotation and filepath list files "File Path mapping tables (XLSX, 49.6 MB)" and "Annotation Boxes (XLSX, 49 kB)" into `data/dbc/maps`. You'll then need to convert these to `.csvs` manually (e.g. using Microsoft Excel).

3. Download the DBC Dataset "Images (DICOM, 368.4 GB)" as follows; unfortunately this is large due to the data only being avaliable as DICOM files. You'll have to use TCIA's NBIA Data Retriever tool for this (open the downloaded `.tcia` manifest file with the tool). **Make sure that you download the files with the "Classic Directory Name" directory type convention.** Otherwise certain files will be mislabeled in the downloaded annotation file, and you'll have to redownload all data from scratch. There are still certain typos in the downloaded annotation files from TCIA, but the subsequent code that we provide has fixes for these.

4. Once all of the data is downloaded, it will be in a folder named `manifest-{...}`, where `{...}` is some auotgenerated string of numbers, for example `manifest-1607053360376`. This folder may be within a subdirectory or two. Move this manifest folder into `data/dbc`.

5. Open the IPython notebook `data/dbc/png_extractor.ipynb`; in cell 2, modify `data_path` in line 4 to be equation to the name of your manifest folder, `manifest-{...}`.

6. Run all cells of this IPython notebook to extract .png images from the raw DICOM files into `data/dbc/png_out`.

7. To create the subset of images that we'll use for experiments, run the IPython notebook `data/dbc/make_subset.ipynb`.

8. Feel free to delete the original DICOM files in your manifest folder once this is complete, as well as `data/dbc/png_out`, to save space.

#### MURA

1. Similar to CheXpert, download the MURA v1.1 dataset from StanfordML [here](https://stanfordmlgroup.github.io/competitions/mura/). You'll need to register in their system and follow their instructions. Once you get an email from them, you can either download the data in browser or from the command line (recommended due to the large size) with `wget --show-progress -O mura.zip {url}`, where `url` is the url that they supply you. Make sure that it saved into a directory named `MURA-v1.1`, i.e. `data/mura/MURA-v1.1`; this directory should have data folders like `train`, `valid` and label `.csvs` like `train_image_paths.csv`.

2. That's it; due to MURA's relatively smaller size, we don't need to create a subset.

#### OAI

#### Prostate-MRI

1. Like DBC, download the Prostate-MRI-US-Biopsy data from The Cancer Imaging Archive [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=68550661). This will take a couple of steps; all files are found under "Data Access" near the bottom of the page.

2. Download the dataset "Images (DICOM) 77.6 (GB)" as follows. You'll have to use TCIA's NBIA Data Retriever tool for this (open the downloaded `.tcia` manifest file with the tool). **Make sure that you download the files with the "Classic Directory Name" directory type convention.** Otherwise certain files will be mislabeled in the downloaded annotation file, and you'll have to redownload all data from scratch. There are still certain typos in the downloaded annotation files from TCIA, but the subsequent code that we provide has fixes for these.

3. Once all of the data is downloaded, it will be in a folder named `manifest-{...}`, where `{...}` is some auotgenerated string of numbers, for example `manifest-1607053360376`. This folder may be within a subdirectory or two. Move this manifest folder into `data/prostate`.

4. Next, download the annotation list file "Target Data (XLSX) 131 (KB)" into `data/prostate/manifest-{...}`. You'll then need to convert this to a `.csv` manually (e.g. using Microsoft Excel).

5. Move the IPython notebook `data/prostate/png_extractor.ipynb` into `data/prostate/manifest-{...}`, and run all cells to extract .png images from the raw DICOM files into `data/prostate/manifest-{...}/train_png`.

6. Move `data/prostate/manifest-{...}/train_png` into `data/prostate/train_png`. You can delete the raw DICOM files from TCIA in the manifest folder now if you'd like, to save space.

#### RSNA-IH-CT

1. The RSNA-IH-CT dataset is from the RSNA Intracranial Hemorrhage Detection Kaggle challenge; follow the instructions [here](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data) to download the dataset (it is large). Move the downloaded file `stage_2_train.csv` and directory `stage_2_train` into `data/rsna`.

2. Run the IPython notebook `data/rsna/png_extractor.ipynb` to convert the downloaded DICOM files into `.png` files. This will take some time. The extracted files will appear in `data/rsna/stage_2_train_png`.

3. Feel free to delete the downloaded files (`stage_2_train`) once the extraction is complete.

### (2) Experiment 1: Compute Intrinsic Dimension of Datasets (Paper Section 4.1)

1. Install the dimension estimate source code by `git clone`ing the repository https://github.com/ppope/dimensions into the base directory (so you will have a new directory called `dimensions`).

2. All dataset intrinsic dimension calculations (results shown in Fig. 2 of the paper) can be completed in the IPython notebook `dimension_calc.ipynb`, as follows. You may need to specify device (GPU/CPU) usage for PyTorch on lines 15-19 of cell 2.

3. Specify the dataset (named according to the directories in `data`, e.g. `prostate` or `rsna`) on line 2 of cell 3.

4. Run all cells. All results will be logged automatically in `logs/dimensionality`; the estimated IDs will be for k=20, using the default MLE estimator.

### (3) Experiment 2: Generalization Ability vs Intrinsic Dimension (Paper Section 4.2)

1. All experiments can be run in the notebook `generalization.ipynb`, as follows. You may need to specify device (GPU/CPU) usage for PyTorch on lines 17-21 of cell 1.

2. Run all cells as they are to reproduce the experiments for everything but "Dependence on Task Choice" (see the following item). This will cover all datasets, models and training set sizes. All training and evaluation results will be logged automatically in `logs/generalization`, and models will be saved in `saved_models/generalization`.

3. To reproduce the "Dependence on Task Choice" results, the procedure is exactly the same, except the list `labelings` on line 19 of cell 2, and the list `dataset_names` on line 4, must be modified, as follows. To test for detecting edema rather than pleural effusion on CheXpert, change line 4 to `dataset_names = ['chexpert']` and line 19 to `labelings = ['Edema']`, and restart the kernel and run all cells. To test for detecting subarachnoid hemorrhage on RSNA-IH-CT instead of any hemorrhage, change line 4 to `dataset_names = ['rsna']` and line 19 to `labelings = ['subarachnoid']`, and restart the kernel and run all cells. Finally, to test detecting severe cancer rather than any cancer on Prostate-MRI, change line 4 to `dataset_names = ['prostate']` and line 19 to `labelings = ['hard']`, and restart the kernel and run all cells.