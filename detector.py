
import tensorflow as tf
from model import NeuralModel
import numpy as np

import argparse
from utils import Utils
#from keras.preprocessing.sequence import pad_sequences

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="'1' to preprocess the original data or '2' to train the model (no quotes)")
parser.add_argument("--dataset", help="Choose 'byarticles' or 'bypublisher' (no quotes)")
arg = parser.parse_args()

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("BERT_MAX_LEN", 512,
	"Bert max sequence len")
tf.app.flags.DEFINE_integer("num_samples", 0,
	"Number of samples (texts)")
tf.app.flags.DEFINE_integer("batch_size", 32,
	"Batch size for training")
tf.app.flags.DEFINE_integer("num_epochs", 12,
	"Number of training epochs")
tf.app.flags.DEFINE_integer("num_folds", 5,
	"Number of k-folds")
tf.app.flags.DEFINE_integer("num_classes", 2,
	"Number of classes ('hyperpartisan' or 'not hyperpartisan'")
tf.app.flags.DEFINE_integer("num_features", 768,
	"Number of features from BERT model")

tf.app.flags.DEFINE_string("checkpoint_folder", "checkpoints", 
	"Where the pre-built embeddings are saved")
tf.app.flags.DEFINE_string("data_folder", "processed_data",
	"Folder where the preprocessed data is saved")
tf.app.flags.DEFINE_string("figure_folder", "figures",
	"Folder where the figures will be saved")

# Workaround
tf.app.flags.DEFINE_string("mode", arg.mode,
	"Workaround")
tf.app.flags.DEFINE_string("dataset", arg.dataset,
	"Workaround")

def main():
	utils = Utils(dataset=arg.dataset, FLAGS=FLAGS)

	# Loading processed texts
	X, Y = utils.load("processed_data/")

	FLAGS.num_samples = len(X)

	#embeddings = utils.bert_as_service_embedding(X[:2])

	if arg.mode == "1":
		""" This will build and checkpoint the embeddings"""

		# Tokenize, pad and convert to numerical representations
		utils.preprocessing(X)

	 	# Saves the preprocessed to a file
		utils.save_preprocessed(X)

	elif arg.mode == "2":
		""" This will train the model on the pre-built embeddings"""
		
		# E(X) is the embedding of X (input data)
		X = utils.load_preprocessed()

		model = NeuralModel(FLAGS)

		# Give as input X embedding (E(X)) and Y
		model.train(X, Y)

if __name__ == "__main__":
	main()
