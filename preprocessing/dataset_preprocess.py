"""
loop through a directory and extarct the text of each file and make a list of the text
"""
from fileinput import filename
import os
import re
import nltk
import gensim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def preprocess(data_dir):
    word_list = []
    for directory in os.listdir(data_dir):
        #print(directory)
        for file in os.listdir(data_dir+'/'+ directory):
            #print(file)
            filename=os.path.join(data_dir+'/'+ directory, file)
            # print(filename)
            with open(filename,'r') as f:
                data=f.readlines()
            # print(data)
            # remove punctuation
            # remove \n in the text
            # remove stopwords
            data = [re.sub(r'[^\w\s]','',line) for line in data]
            data = [re.sub(r'\n','',line) for line in data]
            data = [re.sub(r'\s+',' ',line) for line in data]
            data = [line.lower() for line in data]
            data = [word for word in data if word not in stopwords.words('english')]
            

            # if the len of a word is 1, remove it
            data = [word for word in data if len(word)>1]
            # remove space in the text
            data = [word.strip() for word in data]
            # print(data)
            word_list.extend(data)
            # print((word_list))
          
           

            # text_tokenizer=list(map(word_tokenize,data.split()))
            # print(text_tokenizer)
            
    return word_list



if __name__ == '__main__':
    dataset_dir="D:/Code and Tutorial Practice/NSL_work/Gensim_doc2vec/datasets/20news-bydate-test"
    word_list=preprocess(dataset_dir)
    # print(word_list)