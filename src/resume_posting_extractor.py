import os
import re
import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords    
import nltk
from cover_letter_datareader import CoverLetterDataset 

nltk.download('punkt')

class Parser():
    def __init__(self):
        self.trainData = pd.read_csv('data/cover-letter-dataset/train.csv')

    def skillsetParse(self, resume):
        trainingSkillData = self.trainData["Skillsets"].str.cat(sep=' ')
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3), max_features=10)
        vectors = vectorizer.fit_transform([trainingSkillData, resume])
        feature_names = vectorizer.get_feature_names_out() 

        skillset = []
        for col, term in enumerate(feature_names):
            skillset.append(term)

        return skillset

    def experienceParse(self, resume):
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(7,12), max_features=10)
        trainingExperienceData = self.trainData["Past Working Experience"].str.cat(sep=' ') 
        trainingExperienceData += self.trainData["Current Working Experience"].str.cat(sep=' ')

        vectors = vectorizer.fit_transform([trainingExperienceData, resume])
        feature_names = vectorizer.get_feature_names_out()
        experience = []
        for col, term in enumerate(feature_names):
            experience.append(term)

        return experience

    def qualificationsParse(self, resume, posting):
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(2,12), max_features=10)
        trainingQualificationData = self.trainData["Preferred Qualifications"].str.cat(sep=' ') 

        vectors = vectorizer.fit_transform([trainingQualificationData, resume, posting['description']])
        feature_names = vectorizer.get_feature_names_out()
        qualifications = []
        for col, term in enumerate(feature_names):
            qualifications.append(term)

        return qualifications

    def getCompany(self, posting):
        companyData = pd.read_csv('data/job-posting-dataset/companies.csv')
        company = companyData.loc[companyData['company_id']==posting['company_id']]
        companyName = company['name'].to_string(index=False)
        return companyName

    def getTitle(self, posting):
        return posting['title']


if __name__ == '__main__':
    resumeData = pd.read_csv('data/resume-dataset/Resume/Resume.csv')
    resumeData = resumeData.loc[resumeData['Category']=='INFORMATION-TECHNOLOGY']

    postingData = pd.read_csv('data/job-posting-dataset/job_postings.csv')
    companyData = pd.read_csv('data/job-posting-dataset/company_industries.csv')
    postingData = postingData.join(companyData.set_index('company_id'), on='company_id')
    postingData = postingData[['description','industry','company_id','title']].loc[postingData['industry']=='Information Technology & Services']

    parser = Parser()
    skills = parser.skillsetParse(resumeData.iloc[3]['Resume_str'])
    print(skills)
    experience = parser.experienceParse(resumeData.iloc[3]['Resume_str'])
    print(experience)
    qualifications = parser.qualificationsParse(resumeData.iloc[3]['Resume_str'], postingData.iloc[3])
    print(qualifications)
    companyName = parser.getCompany(postingData.iloc[3])
    print(companyName)
    jobTitle = parser.getTitle(postingData.iloc[3])
    print(jobTitle)

    