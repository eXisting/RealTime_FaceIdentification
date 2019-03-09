from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

import data_preprocess
from classifier import training

datadir = './data/feed'
modeldir = './model/20170511-185253.pb'
logsDir = './logs'
classifier_filename = './classifier/classifier.pkl'

data_preprocess.processFeedData()

print ("Start training")
obj = training(datadir, modeldir, classifier_filename, logsDir)
get_file = obj.main_train()

print('Saved classifier model to file "%s"' % get_file)

sys.exit("Classifier hsa been trained successfuly!")
