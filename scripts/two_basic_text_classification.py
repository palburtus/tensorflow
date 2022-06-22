import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

class TextClassifier:
    def run(self):
        print("\nClassifying Text")
        
        #download text reviews dataset from IMDB
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

        dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                            untar=True, cache_dir='.',
                                            cache_subdir='')

        dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
        os.listdir(dataset_dir)
        
        #read training dir from file
        train_dir = os.path.join(dataset_dir, 'train')
        os.listdir(train_dir)
        
        #open and print a single text file
        sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
        with open(sample_file) as f:
            print(f.read())