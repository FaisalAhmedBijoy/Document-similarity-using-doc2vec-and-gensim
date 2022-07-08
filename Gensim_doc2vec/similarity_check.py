import gensim
import numpy as np
from preprocessing.dataset_preprocess import preprocess
from nltk.tokenize import word_tokenize
from gensim import similarities
from sklearn.metrics.pairwise import cosine_similarity

def vector_reshape_and_tokenization(model,test_data):
    test_data=test_data.lower()
    test_data = word_tokenize(test_data)
    print("test_data",test_data)
    vector = model.infer_vector(test_data)
    # reshape the vector to 1D array
    vector=vector.reshape(-1,1)
    # print("V2_reshape", v2)
    # tokenization
    return vector 

def similarity_check(model_path,vector_path,dataset_dir):
    # load the model
    model = gensim.models.Doc2Vec.load(model_path)
    # load the vector of the model
    word_vector=model.wv.load_word2vec_format(vector_path,binary=False)
    # word_list=preprocess(dataset_dir)
    # test_data_1='i love chat'
    # test_data_2='I love chating'
    # v1=vector_reshape_and_tokenization(model,test_data_1)
    # v2=vector_reshape_and_tokenization(model,test_data_2)
  
    test_data_1 = word_tokenize('i love chat'.lower())
    v1 = model.infer_vector(test_data_1)
    print("V1_infer", v1)
    # reshape the vector to 1D array
    v1 = v1.reshape(1, -1)
 

    test_data_2 = word_tokenize("I love chatting".lower())
    v2 = model.infer_vector(test_data_2)
    print("V2_infer", v2)
    # reshape array.reshape(-1,1)
  
    v2=v2.reshape(1, -1)
    # print("V2_reshape", v2)

    # cosine similarity of two vectors
    cosine_sim = cosine_similarity(v1, v2)
    print("Test data 1: ",test_data_1)
    print("Test data 2: ",test_data_2)
    print("Cosine similarity:", cosine_sim)

    
    # for i in range(len(word_list)):
    #     for j in range(len(word_list)):
    #         if i!=j:
    #             print(i,j,model.wv.similarity(str(i),str(j)))

if __name__ == '__main__':
    model_path='D:/Code and Tutorial Practice/NSL_work/Gensim_doc2vec/models/doc2vec_model.bin'
    vector_path='D:/Code and Tutorial Practice/NSL_work/Gensim_doc2vec/models/doc2vec_vector.txt'
    dataset_dir="D:/Code and Tutorial Practice/NSL_work/Gensim_doc2vec/datasets/20news-bydate-test"
    similarity_check(model_path,vector_path,dataset_dir)