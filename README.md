# bcell_proj

## Overview
This project aims to develop an AI model to predict B-cell epitopes (linear peptides) in antigen proteins. B-cell epitopes are specific sequences within antigens that can activate B-cells, leading to the production of antibodies to neutralize pathogens. Accurate prediction of these epitopes can accelerate vaccine and therapeutic development. In this repo, we exploited the ESM-1b model [[Paper](https://www.pnas.org/doi/10.1073/pnas.2016239118#:~:text=https%3A//doi.org/10.1073/pnas.2016239118)]. 

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

## Usage
### Training
To train the model, use the following command:
```
python main.py --mode train
```
### Inference
To run inference using the trained model:
```
python main.py --mode inference
```
### Notice
- If you want to tune the model's hyperparameters, modify the config.py[https://github.com/ezzy4me/bcell_proj/blob/main/config.py] file.
- Ensure the paths for data and model loading are correctly specified.
- For further customization or issues, please refer to the script comments and modify as necessary!

## Visualization
Additionally, you can visualize attention maps for protein sequence data using a finetuned language model, as shown in the image below and more. The script processes the input data, generates attention maps for specified samples, and saves the visualizations to the designated path.
#### Example of attention map visualization
```
python visualization.py --num_samples 10 --max_length 64 --save_path /home/juhwan/sangmin/bcell_active/attention_maps
```
