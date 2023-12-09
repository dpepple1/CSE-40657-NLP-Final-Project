import os
import re
import pandas as pd
from bs4 import BeautifulSoup

def get_data(data_dir, data_file, other_data_file):
    path = os.path.join(data_dir, data_file)
    df = pd.read_csv(path)
    path2 = os.path.join(data_dir, other_data_file)
    df2 = pd.read_csv(path2)
    df = df.join(df2.set_index('company_id'), on='company_id')
    return df
    
if __name__ =='__main__':
    df = get_data('data/job-posting-dataset/','job_postings.csv','company_industries.csv')
    #df = cleanse_html(df)

    print(df.columns)
    print(df[['description','industry']].loc[df['industry']=='Information Technology & Services'])
