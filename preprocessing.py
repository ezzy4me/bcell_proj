import pandas as pd
from tqdm import tqdm
from config import CFG


def get_preprocessing(data_type, new_df):
    print(f'{data_type} dataframe preprocessing...')
    print('epitope_seq:', new_df['epitope_seq'][0])
    print('antigen_seq:', new_df['antigen_seq'][0])
    print('start and end position:', new_df['start_position'][0], new_df['end_position'][0])

    epitope_list = new_df['epitope_seq'].tolist()
    left_antigen_list = []
    right_antigen_list = []

    for antigen, s_p, e_p in tqdm(zip(new_df['antigen_seq'], new_df['start_position'], new_df['end_position'])):
        start_position = s_p - CFG['ANTIGEN_WINDOW'] - 1
        end_position = e_p + CFG['ANTIGEN_WINDOW']
        if start_position < 0:
            start_position = 0
        if end_position > len(antigen):
            end_position = len(antigen)

        left_antigen = antigen[int(start_position): int(s_p) - 1]
        right_antigen = antigen[int(e_p): int(end_position)]

        left_antigen_list.append(left_antigen)
        right_antigen_list.append(right_antigen)

    label_list = None  # label_list : only exist in train, valid dataframe
    if data_type != 'test':
        label_list = new_df['label'].tolist()

    print(f'{data_type} dataframe preprocessing was done.')
    return epitope_list, left_antigen_list, right_antigen_list, label_list
