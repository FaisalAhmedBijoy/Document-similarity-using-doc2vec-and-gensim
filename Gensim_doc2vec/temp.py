import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from preprocessing.dataset_preprocess import preprocess
def build_model(train_docs, test_docs, comp_docs):
    '''
    Parameters
    -----------
    train_docs: list of lists - combination of known both sentence list
    test_docs: list of lists - one of the sentence lists
    comp_docs: list of lists - combined sentence lists to match the index to the sentence 
    '''
    # Train model
    model = Doc2Vec(dm = 0, dbow_words = 1, window = 2, alpha = 0.2)#, min_alpha = 0.025)
    model.build_vocab(train_docs)
    for epoch in range(10):
        model.train(train_docs, total_examples = model.corpus_count, epochs = epoch)
        #model.alpha -= 0.002
        #model.min_alpha = model.alpha


    scores = []

    for doc in test_docs:
        dd = {}
        # Calculate the cosine similarity and return top 40 matches
        score = model.docvecs.most_similar([model.infer_vector(doc)],topn=40)
        key = " ".join(doc)
        for i in range(len(score)):
            # Get index and score
            x, y = score[i]
            #print(x)
            # Match sentence from other list
            nkey = ' '.join(comp_docs[x])
            dd[nkey] = y
        scores.append({key: dd})

    return scores
if __name__ == '__main__':
    dataset_path='D:/Code and Tutorial Practice/NSL_work/Gensim_doc2vec/datasets/20news-bydate-test'
    train_docs = preprocess(dataset_path)
    test_docs = preprocess(dataset_path)
    comp_docs = preprocess(dataset_path)
    scores = build_model(train_docs, test_docs, comp_docs)
    print(scores)