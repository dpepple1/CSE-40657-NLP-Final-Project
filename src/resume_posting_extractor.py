import os
import re
import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from cover_letter_datareader import CoverLetterDataset 
import random
import spacy
from sklearn.metrics.pairwise import cosine_similarity


class Parser():
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        self.trainData = pd.read_csv('data/cover-letter-dataset/train.csv')

    def skillsetParse(self, resume):
        trainingSkillData = set(self.trainData["Skillsets"].str.cat(sep=' ').split(","))
        trainingSkillData = " ".join(trainingSkillData)
        print(len(trainingSkillData))
        resume = " ".join(set(resume.split()))
        doc = self.nlp(resume)
        
        similarities = {}
        for word in trainingSkillData:
            tok = self.nlp(word)
            similarities[tok.text] = {}
            for token in doc:
                similarities[tok.text].update({token.text:tok.similarity(token)})

        top2 = lambda x: {k: v for k, v in sorted(similarities[x].items(), key=lambda item: item[1], reverse=True)[:2]}
        for word in words:
            print(top2(word))
        exit()
        # trainingSkillData = self.trainData["Skillsets"].dropna()
        # vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3), max_features=50)
        # vectors = vectorizer.fit_transform(trainingSkillData)
        # feature_names = vectorizer.get_feature_names_out()
        # res_vectors = vectorizer.fit_transform([resume])
        # res_features = vectorizer.get_feature_names_out()
        # print(set(feature_names).intersection(set(res_features)))

        return res_features

    def experienceParse(self, resume):
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(6,11), max_features=10)
        trainingExperienceData = self.trainData["Past Working Experience"].str.cat(sep=' ') 
        trainingExperienceData += self.trainData["Current Working Experience"].str.cat(sep=' ')

        vectors = vectorizer.fit_transform([resume])
        feature_names = vectorizer.get_feature_names_out()
        
        feature_names = " ".join(feature_names)

        return feature_names

    def qualificationsParse(self, resume, posting):
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(2,9), max_features=5)
        trainingQualificationData = self.trainData["Preferred Qualifications"].str.cat(sep=' ') 

        vectorizer.fit_transform([resume, posting['description']])
        feature_names = vectorizer.get_feature_names_out()
        
        feature_names = " ".join(feature_names)

        return feature_names

    def getCompany(self, posting):
        companyData = pd.read_csv('data/job-posting-dataset/companies.csv')
        company = companyData.loc[companyData['company_id']==posting['company_id']]
        companyName = company['name'].to_string(index=False)
        return companyName

    def getTitle(self, posting):
        return posting['title']

    def getName(self):
        names = ['James Smith','Michael Smith', 'Robert Smith','Maria Garcia','Maria Rodriguez',
                'David Smith','Mary Smith','Maria Hernandez','Maria Martinez','James Johnson']
        index = random.randint(0,len(names)-1)
        return names[index]

if __name__ == '__main__':
    resumeData = pd.read_csv('data/resume-dataset/Resume/Resume.csv')
    resumeData = resumeData.loc[resumeData['Category']=='INFORMATION-TECHNOLOGY']

    postingData = pd.read_csv('data/job-posting-dataset/job_postings.csv')
    companyData = pd.read_csv('data/job-posting-dataset/company_industries.csv')
    postingData = postingData.join(companyData.set_index('company_id'), on='company_id')
    postingData = postingData[['description','industry','company_id','title']].loc[postingData['industry']=='Information Technology & Services']

    parser = Parser()
    print(resumeData.iloc[9]['Resume_str'])
    skills = parser.skillsetParse(resumeData.iloc[16]['Resume_str'])
    print(skills)
    experience = parser.experienceParse(resumeData.iloc[12]['Resume_str'])
    print(experience)
    qualifications = parser.qualificationsParse(resumeData.iloc[12]['Resume_str'], postingData.iloc[3])
    print(qualifications)
    companyName = parser.getCompany(postingData.iloc[3])
    print(companyName)
    jobTitle = parser.getTitle(postingData.iloc[3])
    print(jobTitle)
    name = parser.getName()
    print(name)
    