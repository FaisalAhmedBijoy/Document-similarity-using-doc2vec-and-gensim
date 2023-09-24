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

def doc2vec(train_word_list, test_word_list, model_save_path, vector_save_path):
    # Create tagged documents for training data
    tagged_train_docs = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(train_word_list)]
    
    # Create tagged documents for test data
    tagged_test_docs = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(test_word_list)]
    
    # Initialize Doc2Vec model
    model = Doc2Vec(vector_size=300, window=5, min_count=5, workers=4, epochs=20)
    
    # Build vocabulary from training data
    model.build_vocab(tagged_train_docs)
    
    # Train the model on training data
    model.train(tagged_train_docs, total_examples=model.corpus_count, epochs=model.epochs)
    
    # Save the trained model
    model.save(model_save_path)
    
    # Generate document vectors for training and test data
    train_vectors = [model.infer_vector(doc.words) for doc in tagged_train_docs]
    test_vectors = [model.infer_vector(doc.words) for doc in tagged_test_docs]
    
    # Save the document vectors
    np.save(vector_save_path + '/train_vectors.npy', train_vectors)
    np.save(vector_save_path + '/test_vectors.npy', test_vectors)
    print('doc2vec model saved at ',model_save_path)

if __name__ == '__main__':
    train_dataset_dir="datasets/20news-bydate-train"
    test_dataset_dir="datasets/20news-bydate-test"
    model_save_path="models/doc2vec_model.model"
    vector_save_path="models/doc2vec_vector.txt"

    train_word_list=preprocess(train_dataset_dir)
    test_word_list=preprocess(test_dataset_dir)
    print('test word list ',test_word_list)

    doc2vec(train_word_list,test_dataset_dir,model_save_path,vector_save_path)
   
