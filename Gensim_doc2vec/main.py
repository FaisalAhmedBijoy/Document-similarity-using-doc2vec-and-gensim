"""
doc2vec using gensim library
"""
from pyexpat import model
import gensim
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from preprocessing.dataset_preprocess import preprocess

def doc2vec(word_list):
  
   

    """
    TaggedDocument is a class that is used to create a corpus of documents.
    """
    tagged_documents = [TaggedDocument(words=word_list, tags=[str(i)]) for i, word_list in enumerate(word_list)]    
    # print("TaggedDocuments",tagged_documents)
    
    model=Doc2Vec(documents=tagged_documents,vector_size=100,window=5,min_count=1,workers=4)
    model.build_vocab(tagged_documents)

    # training the model
    model.train(tagged_documents,total_examples=model.corpus_count,epochs=2)
    model.save("D:/Code and Tutorial Practice/NSL_work/Gensim_doc2vec/models/doc2vec_model.bin")
    # save the vector of the model
    model.wv.save_word2vec_format("D:/Code and Tutorial Practice/NSL_work/Gensim_doc2vec/models/doc2vec_vector.txt",binary=False)

if __name__ == '__main__':
    dataset_dir="D:/Code and Tutorial Practice/NSL_work/Gensim_doc2vec/datasets/20news-bydate-test"

    word_list=preprocess(dataset_dir)
    doc2vec(word_list)
   
