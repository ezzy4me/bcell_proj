# bcell_proj

## Overview
This project aims to develop an AI model to predict B-cell epitopes (linear peptides) in antigen proteins. B-cell epitopes are specific sequences within antigens that can activate B-cells, leading to the production of antibodies to neutralize pathogens. Accurate prediction of these epitopes can accelerate vaccine and therapeutic development. In this repo, we exploited the ESM-1b model [[Paper](https://www.pnas.org/doi/10.1073/pnas.2016239118#:~:text=https%3A//doi.org/10.1073/pnas.2016239118)]. 

### Model architecture
![Screen Shot 2024-06-04 at 12 14 30 AM](https://github.com/ezzy4me/bcell_proj/assets/87761061/ec65504b-3be9-4b8b-9f2e-77f9b1dd4d36)


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
The script eda.py is designed to calculate specific quantiles (e.g., 0.1, 0.5, 0.9) for epitope, left antigen, and right antigen lengths and provides visualizations to help understand the data distribution.
- Quantile Statistics: Computes the specified quantiles for the lengths of epitopes, left antigens, right antigens, and total lengths, grouped by disease type.
- Data Description: Generates and saves descriptive statistics of the data.
- Plots: Overall Length Distribution, Epitope and Total Length Distribution, Length by Disease Type, Length Comparison, Violin Plots

To use it:
```
python eda.py --quantiles 0.1 0.5 0.9
```
#### Example of disease_type statistics for quantile 0.5
<img width="894" alt="Screen Shot 2024-06-03 at 6 42 23 PM" src="https://github.com/ezzy4me/bcell_proj/assets/87761061/ae376465-03b8-426d-8816-55a2127e0f46">

#### Example of total Total Length Distribution by disease type
![total_length_by_disease_type](https://github.com/ezzy4me/bcell_proj/assets/87761061/eab7bea8-18e0-4df4-b39f-9d656d0ad774)

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
```
python visualization.py --num_samples 10 --max_length 64 --save_path /home/juhwan/sangmin/bcell_active/attention_maps
```
#### Example of attention map visualization
In the image below, the x-axis represents l_antigen, epitope (indicated in red), and r_antigen with the origin as the reference point.

![Screen Shot 2024-06-03 at 4 43 08 AM](https://github.com/ezzy4me/bcell_proj/assets/87761061/caaa49c4-2643-436d-8f17-fbe452206c23)

