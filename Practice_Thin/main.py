from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import spatial
list_text = []
import support
import sys, getopt, os, struct
path_train = "../output/Train_Document_Set"
path_test = "../output/Test_Document_Set"
path_train_idf = "../output_process/Train_Document_IDF/"
test_train_idf = "../output_process/Test_Document_IDF/"


ERROR_PATH =u"""
    Tệp Không Tồn Tại hoặc bị lỗi
"""


def bag_of_words_dense(list_text):
    print('    Start handle list text with result is dense')
    result = CountVectorizer()
    return result.fit_transform(list_text).todense()

def bag_of_words_vocabulary_(list_text):
    print('    Start handle list text with result is vocabulary to compare with bm25')
    result = CountVectorizer()
    result.fit_transform(list_text)
    return result.vocabulary_

def TF_IDF_names(list_text):
    print('    Start handle list text with result is names to compare with bm25')
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tf_idf_matrix = tf.fit_transform(list_text)
    feature_names = tf.get_feature_names()

    return feature_names

def TF_IDF_dense(list_text):
    print('    Start handle list text with result is dense')
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tf_idf_matrix = tf.fit_transform(list_text)
    feature_names = tf.get_feature_names()
    dense = tf_idf_matrix.todense()
    return dense

def run(path,avenue,outp):
    check_url = support.check_path(path)
    print('\nCHECK PATH AND APPEND FILE')
    if(check_url):
        list_path = support.add_path_file(path)
        print('\nREAD FILES IN INPUT FOLDER: %s' % path)
        for filess in list_path:
            with open(filess, 'r') as file_input:  
                data = file_input.readline()
                print('    Read file ' + filess)
                data_clean = support.handle_text(data)
                list_text.append(data_clean) 
        print("\nDONE READ FILE AND START WRITE FILE")
        if(avenue == 2):
            data_result = TF_IDF_dense(list_text)
            print('    Start Write TF-IDF file')
            with open(outp+'TF-IDF.txt', 'w') as file_output: 
                for i in range(len(data_result)):
                    j = i+1
                    oupt = 'D' + str(j) +'.txt \n  '
                    temp = " ".join(str(item) for item in data_result[i])
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
    run(path_test,2,test_train_idf)
    run(path_train,2,path_train_idf)
    return

if __name__ == "__main__":
    main(sys.argv[1:])

