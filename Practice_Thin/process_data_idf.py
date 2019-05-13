from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import spatial
list_text = []
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

def run_cosine(cosine_data, measure):
    result = []
    temp1 =""
    output1 = ""
    for i in range(len(cosine_data)):
        x = []    
        for j in range(len(cosine_data)):
            x.append(1 - spatial.distance.cosine(cosine_data[i],cosine_data[j]))
        result.append(x)
    if(result != []):
        with open('./output/' + measure + '_CosSim.txt', 'w') as file_output:
            for i in range(len(result)):
                temp1 = "".join(str(item).ljust(25) for item in result[i])
                temp1 = temp1.replace('[','')
                temp1 = temp1.replace(']','')
                output1 += temp1
                output1 +='\n\n'
                file_output.write(output1) 
            #file_output.write(" ".join(str(item) for item in result))
    print("    Done to compare with cossin")

def run_bm25(bm25_data, measure):
    result = get_bm25_weights(bm25_data, n_jobs=-1)
    temp1 =""
    output1 = ""
    with open('./output/' + measure + '_OkapiBM25.txt', 'w') as file_output: 
        for i in range(len(result)):
            temp1 = "".join(str(item).ljust(25) for item in result[i])
            temp1 = temp1.replace('[','')
            temp1 = temp1.replace(']','')
            output1 += temp1
            output1 +='\n\n'
            file_output.write(output1)
        #file_output.write("".join(str(item).ljust(25) for item in result))
    print("    Done to compare with bm25")

def run(path,avenue,outp,name_file):
    check_url = support.check_path(path)
    print('\nCHECK PATH AND APPEND FILE')
    if(check_url):
        list_path = support.add_path_file(path)
        print('\nREAD FILES IN INPUT FOLDER: %s' % path)
        for filess in list_path:
            with open(filess, 'r') as file_input:  
                data = file_input.readline()
                while data:
                    data_clean = support.handle_text(data)
                    list_text.append(data_clean)
                    data = file_input.readline() 
                print('    Read file ' + filess)
        print("\nDONE READ FILE AND START WRITE FILE")
        if(avenue == 2):
            data_result = TF_IDF_dense(list_text)
            print('    Start Write TF-IDF file')
            with open(outp+name_file, 'w') as file_output: 
                for i in range(len(data_result)):
                    j = i+1
                    oupt = ''
                    temp = " ".join(str(item) for item in data_result[i])
                    temp = temp.replace('[',' ')
                    temp = temp.replace(']',' ')
                    oupt += temp
                    oupt +='\n'
                    file_output.write(oupt)
        print("\nDONE 100%\n")
    else:
        print(ERROR_PATH)
        sys.exit(2)

def main(argv):
    # run(path_test,2,path_test_idf,"TF-IDF.txt")
    run(path_train,2,path_train_idf,"TF-IDF.txt")

    # run(path_test_index,2,path_test_idf,"TF-IDF-index.txt")
    # run(path_train_index,2,path_train_idf,"TF-IDF-index.txt")

    return

if __name__ == "__main__":
    main(sys.argv[1:])

