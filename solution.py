import tarfile
import os
from collections import Counter
import re

def extract_dataset(tar_path, extract_path):
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)

def load_articles(data_dir):
    articles = {}
    for newsgroup in os.listdir(data_dir):
        newsgroup_path = os.path.join(data_dir, newsgroup)
        if os.path.isdir(newsgroup_path):
            articles[newsgroup] = []
            for article_file in os.listdir(newsgroup_path):
                article_path = os.path.join(newsgroup_path, article_file)
                with open(article_path, 'r', errors='ignore') as file:
                    content = file.read()
                   
                    content = preprocess_text(content)
                    articles[newsgroup].append(content)
    return articles

def preprocess_text(text):
    
    text = re.split(r'\n\n', text, maxsplit=1)[-1]
    
    text = re.sub(r'\n-- .*\n', '\n', text)
    
    text = re.sub(r'\n>.*\n', '\n', text)
    return text


def display_statistics(articles):
    for newsgroup, texts in articles.items():
        print(f"Newsgroup: {newsgroup}")
        print(f"Number of articles: {len(texts)}")
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)
        most_common_words = word_counts.most_common(10)
        print("Most common words:", most_common_words)
        print()


tar_path = '/mnt/data/20_newsgroups.tar.gz' 
extract_path = '/mnt/data/20_newsgroups'  


extract_dataset(tar_path, extract_path)
articles = load_articles(extract_path)
display_statistics(articles)
