# Document-similarity-using-doc2vec-and-gensim
Python implementation of a document similarity checking using Doc2Vec.
## File structure
```bash
Document-similarity-using-doc2vec-and-gensim/
├── data/
│   ├── 20news-bydate.tar.gz
│   ├── 20news-bydate-test
│   └── 20news-bydate-train
├── models/
│   ├── doc2vec_model.bin
│   ├── doc2vec_model.model
│   ├── doc2vec_vector.txt
│   └── doc2vec_model.bin.dv.vectors.npy
├── dataset_preprocess.py
├── inference.py
├── README.md
├── requirements.txt
└── train.py
```
- **data/train_data.txt**: Training data file
- **models/doc2vec_model.bin**: Trained Doc2Vec model file
- **models/doc2vec_model.bin.dv.vectors.npy**:  Document vectors file for the trained model
- **README.md**: Project documentation file
- **requirements.txt**: Required Python packages
- **inference.py**: Script to check similarity between two documents
- **train.py**: Script to train the Doc2Vec model


## Installation
Install the dependencies using pip:
```bash
gensim==4.2.0
nltk==3.5
numpy==1.23.1
numpy==1.23.2
pandas==1.2.0
scikit_learn==0.23.2
```
Install the required packages:

```bash
pip install -r requirements.txt
```

## Training the Doc2Vec model
```bash
python train.py 
```

## Inference
Check the similarity between two documents
```bash
python inference.py
```

<!-- 
Test data 1:  ['bird', 'is', 'beautiful']
Test data 2:  ['bird', 'is', 'beautiful']
Cosine similarity: [[0.9203991]] 
-->

