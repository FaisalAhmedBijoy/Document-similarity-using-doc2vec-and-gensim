import gensim
from preprocessing.dataset_preprocess import preprocess
from nltk.tokenize import word_tokenize
def similarity_check():
    # load the model
    model = gensim.models.Doc2Vec.load("D:/Code and Tutorial Practice/NSL_work/Gensim_doc2vec/models/doc2vec_model.bin")
    # load the vector of the model
    word_vector=model.wv.load_word2vec_format("D:/Code and Tutorial Practice/NSL_work/Gensim_doc2vec/models/doc2vec_vector.txt",binary=False)
    # load the dataset
    # dataset_dir="D:/Code and Tutorial Practice/NSL_work/Gensim_doc2vec/datasets/20news-bydate-test"
    # word_list=preprocess(dataset_dir)
    # similiar_doc=model.docvecs.most_similar('O')
    # print(similiar_doc)
    print( word_vector.similarity('home','home'))
 
    
    # similiar_documents=model.docvec_similarity(word_list[0],word_list[1])
    # print(similiar_documents)
    # print(word_list)
    # compare the similarity between the model and the dataset
    # for i in range(len(word_list)):
    #     for j in range(len(word_list)):
    #         if i!=j:
    #             print(i,j,model.wv.similarity(str(i),str(j)))
    test_data = word_tokenize("When your focus is to improve employee performance, its essential to encourage ongoing\
                        dialogue between managers and their direct reports. Some companies encourage supervisors\
                        to hold one-on-one meetings with employees as a way to facilitate\
                        two-way communication.".lower())
    v1 = model.infer_vect
    print(v1)
if __name__ == '__main__':
    similarity_check()