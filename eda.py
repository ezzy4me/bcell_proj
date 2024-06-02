import pandas as pd
import argparse

def main(quantiles):
    path = '/home/juhwan/sangmin/bcell_active/protein_data'
    all_df = pd.read_csv(path + '/train.csv')
    
    # 데이터 프레임에 left_antigen, right_antigen, epitope 길이 추가
    all_df['left_antigen_length'] = all_df.apply(lambda row: len(row['antigen_seq'][:int(row['start_position']) - 1]), axis=1)
    all_df['right_antigen_length'] = all_df.apply(lambda row: len(row['antigen_seq'][int(row['end_position']):]), axis=1)
    all_df['epitope_length'] = all_df['epitope_seq'].apply(len)
    
    # 각 epitope, left_antigen, right_antigen의 평균 길이 계산
    avg_epitope_length = all_df['epitope_length'].mean()
    avg_left_antigen_length = all_df['left_antigen_length'].mean()
    avg_right_antigen_length = all_df['right_antigen_length'].mean()

    print("Average Epitope Length:", avg_epitope_length)
    print("Average Left Antigen Length:", avg_left_antigen_length)
    print("Average Right Antigen Length:", avg_right_antigen_length)
    
    # disease_type 별 통계 계산
    for quantile in quantiles:
        disease_type_stats = all_df.groupby('disease_type').agg({
            'epitope_length': 'mean',
            'left_antigen_length': lambda x, q=quantile: x.quantile(q),
            'right_antigen_length': lambda x, q=quantile: x.quantile(q)
        }).reset_index()
        
        print(f"\nDisease Type Statistics for quantile {quantile}:")
        print(disease_type_stats)

# python eda.py --quantiles 0.1 0.5 0.9
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute quantile statistics for antigen and epitope lengths.')
    parser.add_argument('--quantiles', nargs='+', type=float, required=True, help='List of quantiles to compute, e.g., 0.1 0.5 0.9')
    args = parser.parse_args()
    
    main(args.quantiles)
