"""
This is the ingestion.py procedure.
Author: Sanggyu Biern
Date: 9th Apr. 2023
"""
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
final_data_name = config['final_data_name']
ingested_files_name = config['ingested_files_name']

final_data_path = os.path.join(output_folder_path, final_data_name)
ingested_files_path = os.path.join(output_folder_path, ingested_files_name)

#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    df = pd.DataFrame()
    file_list = []
    
    for file in os.listdir(input_folder_path):
        if file.endswith('csv'):
            file_path = input_folder_path +'/'+file
            df_temp = pd.read_csv(file_path)
            df = pd.concat([df, df_temp], axis=0, ignore_index=True)
            file_list.append(file_path +'\n')
        
    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)
        
    df = df.drop_duplicates()
    df.to_csv(final_data_path, index=False)
    
    with open(ingested_files_path, 'w') as f:
        f.writelines(file_list)
    



if __name__ == '__main__':
    merge_multiple_dataframe()
