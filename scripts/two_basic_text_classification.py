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

        print("\nDownload text reviews dataset from Stackoverflow")
       
        dataset = tf.keras.utils.get_file("stack_overflow_16k", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

        dataset_dir = os.path.join(os.path.dirname(dataset), '.')
        print(os.listdir(dataset_dir))
        
        #read training dir from file
        train_dir = os.path.join(dataset_dir, 'train')
        os.listdir(train_dir)
        
        
        #open and print a single text file
        sample_file = os.path.join(train_dir, 'java/1.txt')
        
        print("\nReading sample Java file: ", sample_file)
        
        with open(sample_file) as f:
            print(f.read())
        
        #Create a validation set using an 80:20 split of the training data
        batch_size = 32
        seed = 42
        raw_train_ds = tf.keras.utils.text_dataset_from_directory(
            './train', 
            batch_size=batch_size, 
            validation_split=0.2, 
            subset='training', 
            seed=seed)
        
        #Iterated over the dataset and pring out a subset of 3 examples 
        #since this is binary classfication labels are either 0 or 1 which are positive and negative reviews
        for text_batch, label_batch in raw_train_ds.take(1):
            for i in range(3):
                print("Review", text_batch.numpy()[i])
                print("Label", label_batch.numpy()[i])
        
        #create a validation dataset
        raw_val_ds = tf.keras.utils.text_dataset_from_directory(
            './train', 
            batch_size=batch_size, 
            validation_split=0.2, 
            subset='validation', 
            seed=seed)     
        
        #and create the test dataset
        raw_test_ds = tf.keras.utils.text_dataset_from_directory(
            './test', 
            batch_size=batch_size)
        
        #function to remove html tags from the data
        def custom_standardization(input_data):
            lowercase = tf.strings.lower(input_data)
            stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
            return tf.strings.regex_replace(stripped_html,
                                            '[%s]' % re.escape(string.punctuation),
                                            '')
        print("\nCreating Vectorize Layer")
        #create text vectoriaztion layer - used to standardize tokenize and verctorize data 
        max_features = 10000
        sequence_length = 250

        vectorize_layer = layers.TextVectorization(
            standardize=custom_standardization,
            max_tokens=max_features,
            output_mode='int',
            output_sequence_length=sequence_length)
        
        print("\nCalling adapt function")
        #call adapt to fit the state of the preprocessing layer to the dataset and cause the model to build an index of string to ints
        #Make a text-only dataset (without labels), then call adapt
        train_text = raw_train_ds.map(lambda x, y: x)
        vectorize_layer.adapt(train_text)
        
        #function to visualize the results of using this layer       
        def vectorize_text(text, label):
            text = tf.expand_dims(text, -1)
            return vectorize_layer(text), label
        
        print("\nPrinting vectorization")
        # retrieve a batch (of 32 reviews and labels) from the dataset
        text_batch, label_batch = next(iter(raw_train_ds))
        first_review, first_label = text_batch[0], label_batch[0]
        print("Review", first_review)
        print("Label", raw_train_ds.class_names[first_label])
        print("Vectorized review", vectorize_text(first_review, first_label))
        
        #each token has been replacted by an integter, here is an example of looking up the corresponding token (string) for given ints
        print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
        print("181 ---> ",vectorize_layer.get_vocabulary()[181])
        print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
        print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))
        
        #final pre-processing step, apply ythe text vectorization we created to train and validate the test dataset
        train_ds = raw_train_ds.map(vectorize_text)
        val_ds = raw_val_ds.map(vectorize_text)
        test_ds = raw_test_ds.map(vectorize_text)
        
        #cache data (these are large datasets don't want a out of memory error) also prefetch makes data preprocessing and model execution async
        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        #create the model (there are 4 classes so dense = 4)
        embedding_dim = 16
        model = tf.keras.Sequential([
            layers.Embedding(max_features + 1, embedding_dim),
            layers.Dropout(0.2),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.2),
            layers.Dense(4)])

        print("Model summary", model.summary())
        
        
        #creat lose function
        #lose function - measurement of how good your mdel is in terms of perdicting the expected outcome
        #change ferom BinaryCrossentropy to SparseCategoricalCrossentropy when going from 2 classes (binary) to more than 2 classes
        #chante metrics=tf.metrics.BinaryAccuracy(threshold=0.0)) to ['accuracy'] as well
        model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])
        
        #train the model
        epochs = 10
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs)
        
        #evaluate the model
        loss, accuracy = model.evaluate(test_ds)

        print("Loss: ", loss)
        print("Accuracy: ", accuracy)
        
        #print the metric names of the model to get different map keys below 
        print("Model, metric names: ", model.metrics_names)
        
        #plot the accuracy and loss over time 
        history_dict = history.history
        history_dict.keys()
        
        #for non-binary change value to acc
        #acc = history_dict['binary_accuracy']
        acc = history_dict['accuracy']
        
        #for non-binary change val_binary_acuracy to val_acc
        #val_acc = history_dict['val_binary_accuracy']
        val_acc = history_dict['val_accuracy']
        
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(acc) + 1)

        #create the loss plot
        # "bo" is for "blue dot"
        plt.plot(epochs, loss, 'bo', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
        
        #create the accuracy plot
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        plt.show()
        
        
        