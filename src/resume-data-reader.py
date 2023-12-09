import pandas as pd
import os
import re
from bs4 import BeautifulSoup

def get_data(data_dir, data_file):
    path = os.path.join(data_dir, data_file)
    df = pd.read_csv(path)
    return df

def cleanse_html(df):
    df['Resume_html'] = df['Resume_html'].apply(lambda x: BeautifulSoup(x, "lxml").text)
    df = df.rename(columns={"Resume_html": "Resume"})
    return df

if __name__ == '__main__':
    df = get_data('data/resume-dataset/Resume', 'Resume.csv')
    #cleanse_html(df)
    befouh = df.iloc[0]['Resume_html']
    df = cleanse_html(df)
    aftah = df.iloc[0]['Resume_str']
    print(aftah)


