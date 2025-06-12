# A2F-DR
This is an implementation of our drug recommendation model A2F-DR in our submitted paper "Let Molecules Speak for Themselves: Atom-to-Fragment Graph Learning for Safe and Accurate Drug Recommendation".
<hr>

## Requirements
Create the environment from environment.yaml.

Follow [FragNet](https://github.com/pnnl/FragNet) to install fragnet.

## Prepare Data
###Create the following directories.

`mkdir -p ./data/raw/mimic-iii`

`mkdir -p ./data/raw/mimic-iv`

`mkdir -p ./data/processed`

###
Get the certificate first, and then download the MIMIC-III and MIMIC-IV datasets.
+ MIMIC-III: https://physionet.org/content/mimiciii/1.4/
+ MIMIC-IV: https://physionet.org/content/mimiciv/

For MIMIC-III, put the four files, ADMISSIONS.csv, DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv, and PROCEDURES_ICD.csv, into directory `./data/raw/mimic-iii`.

For MIMIC-IV, put the four files, admissions.csv, diagnoses_icd.csv, prescriptions_filtered.csv, and procedures_icd.csv, into directory `./data/raw/mimic-iv`.

###
Get other five files, drug-atc.csv, drug-DDI.csv, idx2SMILES.pkl, ndc2atc_level4.csv, and ndc2rxnorm_mapping.txt, from [Carmen](https://github.com/bit1029public/Carmen) and put them into the directory `./data/raw`.

## Process Data
### Molecules
```python
python ./src/molecule_fragmentation.py
```

### MIMIC-III
```python
python ./src/processing_mimic_iii.py
```

### MIMIC-IV
```python
python ./src/processing_mimic_iv.py
```

## Train Model
### MIMIC-III
A2F-DR (β=0.9):
```python
python ./src/run_mimic_iii.py --cuda 0 --epoch 120 --model_name EHR_MOLf_ACC_a0.95_DDI_g0.9 --use_mol_net --mol_net_type 3 --ddi
```

A2F-DR (β=0.7):
```python
python ./src/run_mimic_iii.py --cuda 0 --epoch 120 --model_name EHR_MOLf_ACC_a0.95_DDI_g0.7 --gamma 0.7 --use_mol_net --mol_net_type 3 --ddi
```

### MIMIC-IV
A2F-DR (β=0.9):
```python
python ./src/run_mimic_iv.py --cuda 1 --epoch 120 --model_name FINAL_EHR_MOLf_acc_a0.95_ddi_g0.9
```
A2F-DR (β=0.7):
```python
python ./src/run_mimic_iv.py --cuda 1 --epoch 120 --model_name FINAL_EHR_MOLf_acc_a0.95_ddi_g0.7 --gamma 0.7
```

## Test Model
Add option `--Test` to the training commands.









