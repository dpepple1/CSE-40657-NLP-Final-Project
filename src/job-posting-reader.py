import os
import re
import pandas as pd
from bs4 import BeautifulSoup

def get_data(data_dir, data_file):
    path = os.path.join(data_dir, data_file)
    df = pd.read_csv(path)
    return df

def cleanse_html(df):
    df = df['job_description'].apply(lambda x: BeautifulSoup(x, "lxml").text).apply(lambda x: re.sub("\n","",x))
    return df

if __name__ =='__main__':
    df = get_data('data/job-posting-dataset/','postings.csv')
    df = cleanse_html(df)
    print(df)