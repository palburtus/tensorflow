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
        
        url = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"

        dataset = tf.keras.utils.get_file("stack_overflow_16k", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

        dataset_dir = os.path.join(os.path.dirname(dataset), 'data/stack_overflow16')
        print(os.listdir(dataset_dir))
        
        #read training dir from file
        train_dir = os.path.join(dataset_dir, 'train')
        os.listdir(train_dir)
        '''
        print("\nDownload text reviews dataset from Stackoverflow")
        url = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"

        print("\nGetting dataset directory")
        dataset_dir = os.path.join(os.path.dirname(dataset), 'stack_overflow_16k')
        
        print("\nCopying dataset to object")
        dataset = tf.keras.utils.get_file("stack_overflow_16k", url,
                                            untar=True, cache_dir='.',
                                            cache_subdir='')

        os.listdir(dataset_dir)
        
        #read training dir from file
        train_dir = os.path.join(dataset_dir, 'train')
        os.listdir(train_dir)
        
        #open and print a single text file
        sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
        with open(sample_file) as f:
            print(f.read())'''