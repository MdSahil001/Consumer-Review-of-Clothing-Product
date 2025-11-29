import numpy as np
import os
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

class EmbeddingVectorizer:
    """
    Manages loading of Word2Vec/GloVe and transformation of text to vectors.
    """
    def __init__(self, method='word2vec', path=None, dim=300):
        self.method = method
        self.path = path
        self.dim = dim
        self.model = None
        
    def load_model(self):
        """
        Loads the pre-trained vectors into memory.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Embedding file not found at {self.path}")
            
        print(f"Loading {self.method} model from {self.path}...")
        
        try:
            if self.method == 'word2vec':
                # [18, 20]: Correct syntax for Gensim 4.0+
                self.model = KeyedVectors.load_word2vec_format(
                    self.path, binary=True
                )
            elif self.method == 'glove':
                # [19]: Convert GloVe format to Word2Vec format for Gensim
                tmp_file = self.path + ".w2v_format"
                glove2word2vec(self.path, tmp_file)
                self.model = KeyedVectors.load_word2vec_format(
                    tmp_file, binary=False
                )
                # Cleanup temp file to save space
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            print(f"Successfully loaded {self.method} vectors.")
            
        except Exception as e:
            print(f"Critical Error loading model: {e}")
            raise

    def transform(self, docs):
        """
        Converts a list of tokenized documents into a matrix of averaged vectors.
        Args:
            docs: List of lists of strings (tokens)
        Returns:
            numpy array of shape (n_docs, dim)
        """
        if self.model is None:
            self.load_model()
            
        X = np.zeros((len(docs), self.dim))
        empty_count = 0
        
        for i, tokens in enumerate(docs):
            if not tokens:
                empty_count += 1
                continue
                
            word_vecs =
            for token in tokens:
                if token in self.model:
                    word_vecs.append(self.model[token])
            
            if word_vecs:
                # Average the vectors
                X[i] = np.mean(word_vecs, axis=0)
            else:
                empty_count += 1
                
        print(f"Vectorization complete. {empty_count} documents had no valid vectors.")
        return X