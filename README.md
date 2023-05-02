# EditAS
## Directory Structure
- models: neural models
- scripts: the scripts for conducting experiments

## Prepare Requirements
- Java 8
- Install python dependencies through conda
- Install nlg-eval and setup

```bash
conda env create -f environment.yml 

pip install git+https://github.com/Maluuba/nlg-eval.git@master 
# set the data_path 
nlg-eval --setup ${data_path}
```

## Download Dataset
- Our dataset,trained model and archived can be downloaded from [here](https://www.aliyundrive.com/s/gQ5CNPwdCvR)
- By default, we store the dataset in `../dataset`


## Train
```bash
cd scripts
# 0 for GPU 0
./train_model.sh 0 ResultFile models.updater.CoAttnBPBAUpdater ../dataset
```

## Infer and Evaluate
```bash
cd scripts
./infer_eval.sh 0 ResultsFile models.updater.CoAttnBPBAUpdater ../dataset
```

## Build Vocab Yourself
You can also build the vocabularies by yourself instead of using the one provided with our dataset.


```bash
# download fastText pre-trained model
cd ../dataset
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip cc.en.300.bin.gz

cd scripts
./build_vocab.sh ../../dataset/cc.en.300.bin
```

## Run Baselines
git clone https://github.com/yh1105/Artifact-of-Assertion-ICSE22
```

The results will be placed in the `dataset` directory, and can be evaluated using `EditAS/eval.py`
