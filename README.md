# bcell_proj

## Overview
This project aims to develop an AI model to predict B-cell epitopes (linear peptides) in antigen proteins. B-cell epitopes are specific sequences within antigens that can activate B-cells, leading to the production of antibodies to neutralize pathogens. Accurate prediction of these epitopes can accelerate vaccine and therapeutic development.

## Implement
### Installation
1. Clone the repository:
```
git clone https://github.com/ezzy4me/bcell_proj.git
cd bcell_proj
```
2. Install required packages:
```
pip install -r requirements.txt
```
### Dataset
Place your dataset in the protein_data directory. The directory structure should be as follows: [link](https://dacon.io/competitions/official/235932/data)
```
protein_data/
    ├── train.csv
    ├── test.csv
    ├── sample_submission.csv
```
### Check data
The script eda.py is designed to calculate specific quantiles (e.g., 0.1, 0.5, 0.9) for epitope, left antigen, and right antigen lengths. To use it:
```
python eda.py --quantiles 0.1 0.5 0.9
```

### Usage
#### Training
To train the model, use the following command:
```
python main.py --mode train
```
#### Inference
To run inference using the trained model:
```
python main.py --mode inference
```
