import resume_posting_extractor
import gpt_trainer
import pandas as pd
import random
import cover_letter_datareader as cld

if __name__=="__main__":

    df = cld.get_data('data/cover-letter-dataset', 'train.csv')
    df = cld.add_prompts(df)
    #johns, df = cld.split_johns(df)
    dataset = cld.CoverLetterDataset(df)

    lm = gpt_trainer.gpt.LanguageModel('gpt2', 'cpu')

    test = cld.get_data('data/cover-letter-dataset', 'test.csv')
    test = cld.add_prompts(test)

    gpt_trainer.train(dataset, test, lm, epochs=10, save_model_on_epoch=True)

    resumeData = pd.read_csv('data/resume-dataset/Resume/Resume.csv')
    resumeData = resumeData.loc[resumeData['Category']=='INFORMATION-TECHNOLOGY']

    postingData = pd.read_csv('data/job-posting-dataset/job_postings.csv')
    companyData = pd.read_csv('data/job-posting-dataset/company_industries.csv')
    postingData = postingData.join(companyData.set_index('company_id'), on='company_id')
    postingData = postingData[['description','industry','company_id','title']].loc[postingData['industry']=='Information Technology & Services']

    resume = resumeData.iloc[random.randint(0,len(resumeData))]['Resume_str']
    posting = postingData.iloc[random.randint(0,len(postingData))]

    parser = resume_posting_extractor.Parser()
    skills = parser.skillsetParse(resume)
    # print(skills)
    experience = parser.experienceParse(resume)
    # print(experience)
    qualifications = parser.qualificationsParse(resume, posting)
    # print(qualifications)
    companyName = parser.getCompany(posting)
    # print(companyName)
    jobTitle = parser.getTitle(posting)
    # print(jobTitle)
    name = parser.getName()
    # print(name)
    prompt = cld.build_prompt(experience, qualifications, companyName, jobTitle, skills, name)
    