import numpy as np
from transformers import GPT2Tokenizer, TFGPT2Model
import tensorflow as tf

class GPT:
    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GPT, cls).__new__(cls)
            cls._instance.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            cls._instance.model = TFGPT2Model.from_pretrained('gpt2')
        return cls._instance

    def inp_vector(self, inp):
        if not isinstance(inp, str):
            inp = "NA"
 
        encoded_inp = self.tokenizer(inp, return_tensors='tf')
        out = self.model(encoded_inp)
        vector = out[0].numpy()[0]
 
        feat_vectors = []
        feat_vectors.append(np.mean(vector, axis=0))
        return np.array(feat_vectors)
