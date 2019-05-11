import sys, getopt, os, struct
import re
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer 
from bs4 import BeautifulSoup


my_stopwords = set(stopwords.words('english') +list(punctuation))
ps = PorterStemmer()

def check_path(path):
    if(os.path.isdir(path)):
        return True
    else:
        return False

def add_path_file(path):
    list_path=[]
    for root,dirs,files in os.walk(path):
        for file_input in files:
            print('    Append file: ' + file_input)
            list_path.append(root + "/" +file_input)
    return list_path


def clean_html(text):
  soup = BeautifulSoup(text,'html.parser')
  return soup.get_text()

def remove_special_character(text):
    string = re.sub('[^\w\s]','',text)
    # string = re.sub('s+',' ',string)
    string = string.strip()
    return string

def handle_text(text):
    text = clean_html(text)
    # text = ' '.join(text)
    sents = sent_tokenize(text)
    sents_cleaned = [remove_special_character(s) for s in sents]

    text_sents_join = ''.join(sents_cleaned)
    words = word_tokenize(text_sents_join)
    words = [word.lower() for word in words]
    # words = [word for word in words if word not in my_stopwords]
    words = [ps.stem(word) for word in words]
    result = ' '.join(words)
    return result
    
