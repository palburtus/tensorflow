#https://www.tensorflow.org/tutorials/quickstart/beginner

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

class ImageClassifier:
    
    def run(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist

        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        train_images.shape
        len(train_labels)

        #number of pixels in each image of the fashion_mnist data set
        max_pixel_range = 255.0

        #divide train and test images by the max pixel range because the values need to be scaled to a range of 0 to 1
        train_images = train_images / max_pixel_range
        test_images = test_images / max_pixel_range


        #display first 25 images from the training set 
        '''
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[train_labels[i]])
        plt.show()
        '''

        #layers - basic building block of neural network. Extract representations from data fed into them.  Most of deep learning is chaining layers together

        #create model
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)), #transforms the image from a 2-d array of 28x28 pixels to 1-d array of 784 pixels (it is like unstacking rows in the image and lining them up)
            tf.keras.layers.Dense(128, activation='relu'), 
            tf.keras.layers.Dense(10)
        ])

        #compile model (including its loss function)
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        #train the model
        model.fit(train_images, train_labels, epochs=10)

        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

        print('\nTest accuracy:', test_acc)

        #Test accuracy: 0.8845999836921692 - 

        '''
        Overfitting - Accuracy on test dataset is less than on training data set.  
        Happens when model performs worse on new unseen inputs that it does on training data.

        Overfitted models "memorize" the noise and details in the training set to the point where it negativly impacts performance 
        '''

        probability_model = tf.keras.Sequential([model, 
                                                tf.keras.layers.Softmax()])

        predictions = probability_model.predict(test_images)

        print('\nPrediction 0: ', predictions[0])

        '''
        Prdictions represent model's confidence that the image corresponds to the right classification.  Higher the number higher the confidence 
        Prediction 0:  [3.8661886e-07 2.2555223e-08 1.8779420e-09 5.1740031e-08 2.7574727e-09
        6.4777704e-03 1.9311990e-07 1.2159114e-02 3.1443509e-08 9.8136246e-01]
        '''

        print('\n Prediection 0 highest confidence array index: ', np.argmax(predictions[0]))
        #9 is the highest in predicutions[0], the model is most confident the image belongs to class 9

        #compare that with the test data set 
        print('\nTest labels prediction: ', test_labels[0])
        #test data set is also 9

        #graph the confidence of each class in the model

        #display image with x-axis
        def plot_image(i, predictions_array, true_label, img):
            true_label, img = true_label[i], img[i]
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])

            plt.imshow(img, cmap=plt.cm.binary)

            predicted_label = np.argmax(predictions_array)
            if predicted_label == true_label:
                color = 'blue'
            else:
                color = 'red'

            plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                            100*np.max(predictions_array),
                                            class_names[true_label]),
                                            color=color)

        #display confidence graph
        def plot_value_array(i, predictions_array, true_label):
            true_label = true_label[i]
            plt.grid(False)
            plt.xticks(range(10))
            plt.yticks([])
            thisplot = plt.bar(range(10), predictions_array, color="#777777")
            plt.ylim([0, 1])
            predicted_label = np.argmax(predictions_array)

            thisplot[predicted_label].set_color('red')
            thisplot[true_label].set_color('blue')


        # Plot the first X test images, their predicted labels, and the true labels.
        # Color correct predictions in blue and incorrect predictions in red.
        num_rows = 5
        num_cols = 3
        num_images = num_rows*num_cols
        plt.figure(figsize=(2*2*num_cols, 2*num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2*num_cols, 2*i+1)
            plot_image(i, predictions[i], test_labels, test_images)
            plt.subplot(num_rows, 2*num_cols, 2*i+2)
            plot_value_array(i, predictions[i], test_labels)
            plt.tight_layout()
            plt.show()


        #used the trained model to try to actually make a prediction about an iamge 
        # Grab an image from the test dataset.
        img = test_images[1]

        print(img.shape)

        # Add the image to a batch where it's the only member.
        img = (np.expand_dims(img,0))

        print(img.shape)

        #make the prediction
        predictions_single = probability_model.predict(img)

        print(predictions_single)

        #plot the prediction
        plot_value_array(1, predictions_single[0], test_labels)
        _ = plt.xticks(range(10), class_names, rotation=45)
        plt.show()

        #predict the class of the image (should be 2)
        print('\nclass:', np.argmax(predictions_single[0]))