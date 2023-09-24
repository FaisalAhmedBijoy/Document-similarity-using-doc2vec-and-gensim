"""
doc2vec using gensim library
"""
from pyexpat import model
import gensim
import numpy as np
import pandas as pd
from nltk import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from dataset_preprocess import preprocess

def doc2vec(train_word_list,test_word_list,model_save_path,vector_save_path):
    pass
    
  

 

if __name__ == '__main__':
    train_dataset_dir="datasets/20news-bydate-train"
    test_dataset_dir="datasets/20news-bydate-test"
    model_save_path="models/doc2vec_model.model"
    vector_save_path="models/doc2vec_vector.txt"

    train_word_list=preprocess(train_dataset_dir)
    test_word_list=preprocess(test_dataset_dir)
    doc2vec(train_word_list,test_dataset_dir,model_save_path,vector_save_path)
   
