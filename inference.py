import gensim
import numpy as np
from dataset_preprocess import preprocess
from nltk.tokenize import word_tokenize
from gensim import similarities
from sklearn.metrics.pairwise import cosine_similarity


def similarity_check(model_path,vector_path,dataset_dir,test_data_1,test_data_2):
    # load the model
    model = gensim.models.Doc2Vec.load(model_path)
    # load the vector of the model
    word_vector=model.wv.load_word2vec_format(vector_path,binary=False)


    test_data_1 = word_tokenize(test_data_1.lower())
    v1 = model.infer_vector(test_data_1)
    print("V1_infer", v1)
    # reshape the vector to 1D array
    v1 = v1.reshape(1, -1)
 

    test_data_2 = word_tokenize(test_data_2.lower())
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
    # convert cosine similarity nxn matrix to 1x1 matrix


if __name__ == '__main__':
    model_path='models/doc2vec_model.bin'
    vector_path='models/doc2vec_vector.txt'
    dataset_dir="datasets/20news-bydate-test"
    test_data_1='Bird is beautiful'
    test_data_2='Bird is beautiful'
    similarity_check(model_path,vector_path,dataset_dir,test_data_1,test_data_2)