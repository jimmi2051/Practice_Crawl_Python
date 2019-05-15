from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import spatial
import support
import sys, getopt, os, struct
import pandas as pd 
import json
import re
path_train = "../output/Train_Document_Set"
path_test = "../output/Test_Document_Set"

path_train_index = "../output/Train_Document_Set_Index"
path_test_index = "../output/Test_Document_Set_Index"

path_train_idf = "../output_process/Train_Document_IDF/"
path_test_idf = "../output_process/Test_Document_IDF/"


ERROR_PATH =u"""
    Tệp Không Tồn Tại hoặc bị lỗi
"""

def TF_IDF_dense(list_text):
    print('    Start handle list text with result is dense')
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), max_df=1.0,encoding='utf-8', min_df=1, stop_words='english')
    tf_idf_matrix = tf.fit_transform(list_text)
    feature_names = tf.get_feature_names()
    print(feature_names)
    dense = tf_idf_matrix.todense()
    return dense
def read_file(file_path):
    with open(file_path,'r') as file_input:
        temp_str = file_input.read()
        
        parse_string = temp_str.split(",")
        
        # for item in parse_string:
        #     result.append(int(item))
        del parse_string[-1]
        return parse_string

def run(file_path):
    check_url = support.check_file(file_path)
    list_text = []
    file_idf = "../output_process/TF_IDF.txt"
    print('\nCHECK PATH AND APPEND FILE')
    if(check_url):
        with open(file_path,'r') as file_input:
            data = file_input.read()
            parse_string = data.split(",")
            del parse_string[-1]
            for item in parse_string:
                data_clean = support.handle_text(item)
                list_text.append(data_clean)
        data_result = TF_IDF_dense(list_text)
        print('     Start write TF-IDF file')
        with open(file_idf,'w') as file_output:
            for i in range(len(data_result)):
                j=i+1
                oupt = ''
                temp = "".join(str(item) for item in data_result[i])
                temp = temp.replace('\n',',\n')   
                temp = temp.replace('  ','')        
                temp = temp.replace(' ',',')      
                temp = temp.replace('[','')
                temp = temp.replace(']','')   
                oupt += temp
                oupt +='\n'
                file_output.write(oupt)
        print("\nDONE 100%\n")
    else:
        print(ERROR_PATH)
        sys.exit(2)
def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("</?.*?>"," <> ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text
def get_stop_words(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)
def new_process():
    df_idf=pd.read_json("stackoverflow-data-idf.json",lines=True)

    # print schema
    print("Schema:\n\n",df_idf.dtypes)
    print("Number of questions,columns=",df_idf.shape)
    df_idf['text'] = df_idf['title'] + df_idf['body']
    df_idf['text'] = df_idf['text'].apply(lambda x:pre_process(x))

    #show the first 'text'
    df_idf['text'][2]
    #load a set of stop words
    stopwords=get_stop_words("stop_word.txt")

    #get the text column 
    docs=df_idf['text'].tolist()

    #create a vocabulary of words, 
    #ignore words that appear in 85% of documents, 
    #eliminate stop words
    cv=CountVectorizer(max_df=0.85,stop_words=stopwords)
    word_count_vector=cv.fit_transform(docs)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    # vectorizer = TfidfVectorizer(min_df=1)
    # X = vectorizer.fit_transform(tfidf_transformer)
    # idf = vectorizer.idf_
    # print (dict(zip(vectorizer.get_feature_names(), idf)))

def main(argv):
    # run(path_test,2,path_test_idf,"TF-IDF.txt")
    run("../output/content.txt")
    # new_process()
    # content = read_file("../output/content.txt")
    # title = read_file("../output/title.txt")
    # label = read_file("../output/label.txt")
    # print(str(len(label)))
    # print(content)
    # print("\n",title)
    # print("\n",label)
    # run(path_test_index,2,path_test_idf,"TF-IDF-index.txt")
    # run(path_train_index,2,path_train_idf,"TF-IDF-index.txt")
    return

if __name__ == "__main__":
    main(sys.argv[1:])

