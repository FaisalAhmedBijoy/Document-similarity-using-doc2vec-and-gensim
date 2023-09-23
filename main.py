"""
doc2vec using gensim library
"""
from pyexpat import model
import gensim
import numpy as np
import pandas as pd
from nltk import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from preprocessing.dataset_preprocess import preprocess

def doc2vec(word_list,model_save_path,vector_save_path):
  
   

    """
    TaggedDocument is a class that is used to create a corpus of documents.
    """
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(word_list)]    
    # print("TaggedDocuments",tagged_documents)
    
    model=Doc2Vec(documents=tagged_data,vector_size=100,window=5,min_count=1,workers=4)
    model.build_vocab(tagged_data)

    # training the model
    model.train(tagged_data,total_examples=model.corpus_count,epochs=2)
    model.save(model_save_path)
    # save the vector of the model
    model.wv.save_word2vec_format(vector_save_path,binary=False)

if __name__ == '__main__':
    dataset_dir="D:/Code and Tutorial Practice/NSL_work/Gensim_doc2vec/datasets/20news-bydate-test"
    model_save_path="D:/Code and Tutorial Practice/NSL_work/Gensim_doc2vec/models/doc2vec_model.model"
    vector_save_path="D:/Code and Tutorial Practice/NSL_work/Gensim_doc2vec/models/doc2vec_vector.txt"

    word_list=preprocess(dataset_dir)
    doc2vec(word_list,model_save_path,vector_save_path)
   
