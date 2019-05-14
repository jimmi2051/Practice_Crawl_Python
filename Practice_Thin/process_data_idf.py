from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import spatial
import support
import sys, getopt, os, struct


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
    print(tf_idf_matrix)
    dense = tf_idf_matrix.todense()
    return dense

def run(file_path):
    check_url = support.check_file(file_path)
    list_text = []
    file_idf = "../output_process/TF_IDF.txt"
    print('\nCHECK PATH AND APPEND FILE')
    if(check_url):
        with open(file_path,'r') as file_input:
            data = file_input.readline()
            data_clean = support.handle_text(data)
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

def main(argv):
    # run(path_test,2,path_test_idf,"TF-IDF.txt")
    run("../output/content.txt")

    # run(path_test_index,2,path_test_idf,"TF-IDF-index.txt")
    # run(path_train_index,2,path_train_idf,"TF-IDF-index.txt")

    return

if __name__ == "__main__":
    main(sys.argv[1:])

