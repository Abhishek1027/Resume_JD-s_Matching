import nltk_spacy
import os
import textract
from tabulate import tabulate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text and keywords from a resume
def process_resume(resume_path):
    resume_text = str(textract.process(resume_path), 'UTF-8').lower()
    keywords_resume_spacy = nltk_spacy.spacy_keywords(resume_text)
    keywords_resume_nltk = nltk_spacy.nltk_keywords(resume_text)
    return resume_text, keywords_resume_spacy, keywords_resume_nltk

# Function to calculate similarities
def calculate_similarities(resume_text, keywords_resume_spacy, keywords_resume_nltk, job_description):
    similarities = []

    text = [resume_text, job_description['text']]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text)
    cosine = cosine_similarity(count_matrix)[0][1]

    # Matching with SpaCy keywords
    keywords_jd_spacy = nltk_spacy.spacy_keywords(job_description['text'])
    keywords_matched_spacy = set(keywords_jd_spacy).intersection(keywords_resume_spacy)
    percentage_spacy = len(keywords_matched_spacy) / len(keywords_resume_spacy)

    # Matching with NLTK keywords
    keywords_jd_nltk = nltk_spacy.nltk_keywords(job_description['text'])
    keywords_matched_nltk = set(keywords_jd_nltk).intersection(keywords_resume_nltk)
    percentage_nltk = len(keywords_matched_nltk) / len(keywords_resume_nltk)

    similarities.append([
        job_description['filename'],
        f'{cosine:.2%}',
        f'{percentage_spacy:.2%}',
        f'{percentage_nltk:.2%}',
    ])
    return similarities

# Iterate over each job role
job_roles = os.listdir('data/jd/')
for job_role in job_roles:
    job_role_name = job_role.split('.')[0]  # Get the job role name without the file extension
    jd_file_path = os.path.join('data/jd', job_role)
    
    # Read the job description text
    job_description_text = str(textract.process(jd_file_path), 'UTF-8').lower()
    job_description = {"filename": job_role, "text": job_description_text}
    
    print('=' * 50)
    print(f'Job Description: {job_description["filename"]}')
    print('=' * 50)
    
    # Path to the resume folder corresponding to the job role
    resume_folder = os.path.join('data/resume', job_role_name)

    # Check if the resume folder exists for the given job role
    if os.path.exists(resume_folder):
        resume_files = sorted(os.listdir(resume_folder))
        
        all_similarities = []
        for resume_file in resume_files:
            resume_path = os.path.join(resume_folder, resume_file)
            resume_text, keywords_resume_spacy, keywords_resume_nltk = process_resume(resume_path)
            similarities = calculate_similarities(resume_text, keywords_resume_spacy, keywords_resume_nltk, job_description)
            
            for sim in similarities:
                sim.insert(0, resume_file)  # Insert the resume filename at the beginning
                all_similarities.append(sim)
        
        print(tabulate(all_similarities, headers=[
            'Resume File',
            'JD File',
            'Cosine %',
            'SpaCy %',
            'NLTK %'
        ]))
        print()
    else:
        print(f'No resumes found for job role: {job_role_name}')
        print()
