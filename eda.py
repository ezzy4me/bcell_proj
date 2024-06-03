import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import textwrap
import matplotlib.patches as mpatches

def main(quantiles):
    path = '/home/juhwan/sangmin/bcell_active/protein_data'
    save_path = 'eda_result'
    os.makedirs(save_path, exist_ok=True)
    
    all_df = pd.read_csv(path + '/train.csv')
    
    # 데이터 프레임에 left_antigen, right_antigen, epitope 길이 추가
    all_df['left_antigen_length'] = all_df.apply(lambda row: len(row['antigen_seq'][:int(row['start_position']) - 1]), axis=1)
    all_df['right_antigen_length'] = all_df.apply(lambda row: len(row['antigen_seq'][int(row['end_position']):]), axis=1)
    all_df['epitope_length'] = all_df['epitope_seq'].apply(len)
    all_df['total_length'] = all_df['left_antigen_length'] + all_df['epitope_length'] + all_df['right_antigen_length']
    
    # 각 epitope, left_antigen, right_antigen의 평균 길이 계산
    avg_epitope_length = all_df['epitope_length'].mean()
    avg_left_antigen_length = all_df['left_antigen_length'].mean()
    avg_right_antigen_length = all_df['right_antigen_length'].mean()
    avg_total_length = all_df['total_length'].mean()

    print("Average Epitope Length:", avg_epitope_length)
    print("Average Left Antigen Length:", avg_left_antigen_length)
    print("Average Right Antigen Length:", avg_right_antigen_length)
    print("Average Total Length:", avg_total_length)
    
    # disease_type 별 통계 계산
    for quantile in quantiles:
        disease_type_stats = all_df.groupby('disease_type').agg({
            'epitope_length': 'mean',
            'left_antigen_length': lambda x, q=quantile: x.quantile(q),
            'right_antigen_length': lambda x, q=quantile: x.quantile(q),
            'total_length': lambda x, q=quantile: x.quantile(q)
        }).reset_index()
        
        print(f"\nDisease Type Statistics for quantile {quantile}:")
        print(disease_type_stats)
    
    # 데이터 요약 통계 추출 및 저장
    description = all_df.describe()
    description.to_csv(os.path.join(save_path, 'data_description.csv'))

    # 질병 유형 이름을 줄여서 범례를 생성
    def shorten_label(label, max_length=10):
        if len(label) > max_length:
            return label[:max_length] + '...'
        return label

    disease_types = all_df['disease_type'].unique()
    short_labels = [shorten_label(dt) for dt in disease_types]
    legend_labels = dict(zip(short_labels, disease_types))

    # 시각화
    # 전체 길이 분포 히스토그램
    melted_df = all_df[['left_antigen_length', 'right_antigen_length', 'epitope_length', 'total_length']].melt(var_name='Length_Type', value_name='Length')
    g = sns.FacetGrid(melted_df, col='Length_Type', col_wrap=2, sharex=False, sharey=False)
    g.map(sns.histplot, 'Length', kde=True, bins=20)  # bin 수를 줄여서 x축 간격을 넓힙니다.
    g.fig.suptitle('Overall Length Distribution', y=1.02)
    plt.savefig(os.path.join(save_path, 'overall_length_distribution.png'))
    plt.clf()
    
    # Epitope Length 및 전체 길이 분포
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    sns.histplot(all_df['epitope_length'], kde=True, bins=20)  # bin 수를 줄여서 x축 간격을 넓힙니다.
    plt.title('Epitope Length Distribution')
    
    plt.subplot(1, 2, 2)
    sns.histplot(all_df['total_length'], kde=True, bins=20)  # bin 수를 줄여서 x축 간격을 넓힙니다.
    plt.title('Total Length Distribution')
    
    plt.savefig(os.path.join(save_path, 'epitope_total_length_distribution.png'))
    plt.clf()
    
    # Disease Type 별 평균 Epitope 및 전체 길이
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='disease_type', y='epitope_length', data=all_df)
    plt.title('Epitope Length by Disease Type')
    short_labels = [shorten_label(label.get_text()) for label in ax.get_xticklabels()]
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    plt.legend(handles=[mpatches.Patch(label=f"{short} ({full})") for short, full in legend_labels.items()], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'epitope_length_by_disease_type.png'))
    plt.clf()

    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='disease_type', y='total_length', data=all_df)
    plt.title('Total Length by Disease Type')
    short_labels = [shorten_label(label.get_text()) for label in ax.get_xticklabels()]
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    plt.legend(handles=[mpatches.Patch(label=f"{short} ({full})") for short, full in legend_labels.items()], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'total_length_by_disease_type.png'))
    plt.clf()
    
    # Disease Type 별 epitope와 total length의 분포
    plt.figure(figsize=(14, 10))
    ax = sns.violinplot(x='disease_type', y='epitope_length', data=all_df, scale='width', inner='quartile')
    plt.title('Epitope Length Distribution by Disease Type')
    short_labels = [shorten_label(label.get_text()) for label in ax.get_xticklabels()]
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    plt.legend(handles=[mpatches.Patch(label=f"{short} ({full})") for short, full in legend_labels.items()], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'epitope_length_violin_by_disease_type.png'))
    plt.clf()
    
    plt.figure(figsize=(14, 10))
    ax = sns.violinplot(x='disease_type', y='total_length', data=all_df, scale='width', inner='quartile')
    plt.title('Total Length Distribution by Disease Type')
    short_labels = [shorten_label(label.get_text()) for label in ax.get_xticklabels()]
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    plt.legend(handles=[mpatches.Patch(label=f"{short} ({full})") for short, full in legend_labels.items()], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'total_length_violin_by_disease_type.png'))
    plt.clf()
    
    # Epitope와 left/right sequence 길이 비교
    plt.figure(figsize=(14, 10))
    melted_lengths = all_df[['left_antigen_length', 'right_antigen_length', 'epitope_length']].melt(var_name='Sequence_Type', value_name='Length')
    ax = sns.boxplot(x='Sequence_Type', y='Length', data=melted_lengths)
    plt.title('Comparison of Sequence Lengths')
    plt.savefig(os.path.join(save_path, 'sequence_length_comparison.png'))
    plt.clf()

# python eda.py --quantiles 0.1 0.5 0.9
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute quantile statistics for antigen and epitope lengths.')
    parser.add_argument('--quantiles', nargs='+', type=float, required=True, help='List of quantiles to compute, e.g., 0.1 0.5 0.9')
    args = parser.parse_args()
    
    main(args.quantiles)
