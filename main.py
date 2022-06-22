import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
from scripts.one_basic_image_classification import ImageClassifier
from scripts.two_basic_text_classification import TextClassifier

print("TensorFlow version:", tf.__version__)


import sys
 
arg = sys.argv[1]

if arg == '1':
  ic = ImageClassifier()
  ic.run()  
elif arg == '2':
  tc = TextClassifier()
  tc.run()
else:
  print("\nInvalid Argument run one of the below", arg)
  print("\n 1 : quick start for begginers")



