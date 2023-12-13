import os
import re
import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from cover_letter_datareader import CoverLetterDataset 
import random
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import skills_parsing


class Parser():
    def __init__(self):
        self.nlp = skills_parsing.main()
        self.skill_pattern_path = "data/resume-dataset/Resume/jz_skill_patterns.jsonl"
        self.trainData = pd.read_csv('data/cover-letter-dataset/train.csv')

    def skillsetParse(self, resume):
        resume = skills_parsing.process_resume(resume)
        doc = self.nlp(resume)
        skills = []
        for entity in doc.ents:
            if entity.label_ == "SKILL":
                #print(entity)
                skills.append(entity.text)
        skills = list(set(skills))

        return ", ".join(skills)

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
    #print(resumeData.iloc[9]['Resume_str'])
    skills = parser.skillsetParse(resumeData.iloc[83]['Resume_str'])
    print(skills)
    experience = parser.experienceParse(resumeData.iloc[83]['Resume_str'])
    print(experience)
    qualifications = parser.qualificationsParse(resumeData.iloc[83]['Resume_str'], postingData.iloc[3])
    print(qualifications)
    companyName = parser.getCompany(postingData.iloc[3])
    print(companyName)
    jobTitle = parser.getTitle(postingData.iloc[3])
    print(jobTitle)
    name = parser.getName()
    print(name)
    