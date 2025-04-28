import pandas as pd
import os

def merge_csv_files(directory, output_file):
    all_files = []
    
    # Traverse directory and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                all_files.append(os.path.join(root, file))
    
    df_list = []

    for file_path in all_files:
        df = pd.read_csv(file_path)
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f'Merged {len(all_files)} CSV files into {output_file}')

if __name__ == '__main__':
    directory = './results_expes/expe_test_expe/'  # Adjust this path as needed
    output_file = './results_expes/merged_results.csv'  # Adjust this path as needed
    merge_csv_files(directory, output_file)