from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.layers import Concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.utils import to_categorical
from pytorch_pretrained_bert import BertModel#, BertForMaskedLM
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import torch
import gc
import os

class NeuralModel():
	def __init__(self, FLAGS, bert_model='bert-base-uncased'):
		self.FLAGS = FLAGS
		self.bert_pretrained_model = bert_model
		self.bert_model = BertModel.from_pretrained(self.bert_pretrained_model)
		self.kfold = KFold(n_splits=self.FLAGS.num_folds, shuffle=True)

	def precision(self, y_true, y_pred):
			"""Precision metric.
			Only computes a batch-wise average of precision.
			Computes the precision, a metric for multi-label
			classification of how many selected items are relevant.
			"""
			true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
			predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

			predicted_positives = K.cast(predicted_positives, dtype='float32')

			prec = true_positives/(predicted_positives + K.epsilon())
			return K.get_value(prec)

	def recall(self, y_true, y_pred):
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		predicted_falses = K.sum(K.round(K.clip(y_true, 0, 1)))

		predicted_falses = K.cast(predicted_falses, dtype='float32')

		rec = true_positives / (predicted_falses + K.epsilon())
		return K.get_value(rec)

	def build_embedding(self, texts):
		""" This will build the embeddings for the given set of texts"""

		embeddings = []

		self.bert_model.eval()

		# 'text' is already in numerical representation
		for i, text in enumerate(texts):
			#print("processing {}/{}".format(i+1, len(texts)))
			text_embedding = []

			segments_ids = [1] * len(text)
			text_tensor = torch.tensor([text])
			segment_tensor = torch.tensor([segments_ids])

			with torch.no_grad():
				encoded_layers, _ = self.bert_model(text_tensor, segment_tensor)

			for token_i in range(len(text)):
				hidden_layers = []

				for layer_i in range(len(encoded_layers)):
					embed = encoded_layers[layer_i][0][token_i] # type : tensor
					hidden_layers.append(embed)
				
				text_embedding.append(hidden_layers)

			# This is for word embedding
			"""concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings] # [number_of_tokens, 3072]"""
			summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0).numpy() for layer in text_embedding] # [number_of_tokens, 768]
			
			# Convert nparray to list (for json-file saving)
			summed_last_4_layers = [_.tolist() for _ in summed_last_4_layers]

			embeddings.append(summed_last_4_layers)

			# This is for sentence embedding
			"""sentence_embedding = torch.mean(encoded_layers[11], 1)
			embeddings.append(sentence_embedding)"""

		return embeddings # [num_texts, num_tokens, num_features]

	def build_graph(self):
		""" This will build the neural graph for future use """
		
		# input_layer : (batch_size, num_tokens, num_features)
		# layer1 : (batch_size, num_tokens-kernel+1, num_filters)
		# output_layer : (batch_size, num_tokens, num_classes)

		input_layer = Input(shape=(self.FLAGS.BERT_MAX_LEN, self.FLAGS.num_features)) 

		layer_1 = Conv1D(filters=512, kernel_size=3, activation="relu")(input_layer)
		layer_1 = MaxPooling1D()(layer_1)
		layer_1 = Dropout(0.5)(layer_1)
		layer_1 = Flatten()(layer_1)

		layer_2 = Conv1D(filters=512, kernel_size=4, activation="relu")(input_layer)
		layer_2 = MaxPooling1D()(layer_2)
		layer_2 = Dropout(0.5)(layer_2)
		layer_2 = Flatten()(layer_2)

		layer_3 = Conv1D(filters=512, kernel_size=5, activation="relu")(input_layer)
		layer_3 = MaxPooling1D()(layer_3) 
		layer_3 = Dropout(0.5)(layer_3)
		layer_3 = Flatten()(layer_3)

		concat_layer = Concatenate()([layer_1, layer_2, layer_3])

		concat_layer = Dropout(0.5)(concat_layer)

		output_layer = Dense(self.FLAGS.num_classes, activation='softmax')(concat_layer)

		return Model([input_layer], output_layer)

	def train(self, X, Y):
		""" This will train the built graph """

		# Convert X to nparray and Y to one-hot representation
		X = np.array([np.array(x) for x in X])
		Y = to_categorical(Y)

		# Build model
		model = self.build_graph()
		model.summary()
		#model.compile(optimizer='rmsprop',loss='binary_crossentropy',
		#        metrics=[categorical_accuracy])
		model.compile(optimizer='rmsprop',loss='categorical_crossentropy',
		        metrics=["accuracy"])
		
		fold = 0

		training_losses, validation_losses = [], []

		# K-fold 
		for train_index, val_index in self.kfold.split(X):
			print("Fold {} of {}".format(fold+1, self.FLAGS.num_folds))
			fold += 1

			# Data is splitted in 80:20 for train/validation
			x_train, y_train, x_val, y_val = X[train_index], Y[train_index], X[val_index], Y[val_index]

			# Train the model
			for epoch in range(self.FLAGS.num_epochs):
				print("Epoch {} of {}".format(epoch+1, self.FLAGS.num_epochs))
				
				i = 0
				while i < x_train.shape[0]:
					start = i
					end = start + self.FLAGS.batch_size
					batch_size = self.FLAGS.batch_size

					if end > self.FLAGS.num_samples:
					    end = self.FLAGS.num_samples
					    batch_size = end - start

					E_X = self.build_embedding(x_train[start:end])
					E_X = np.array([np.array(sample) for sample in E_X])

					loss = model.train_on_batch([E_X], y_train[start:end])
					
					# Calculate precision
					ypred = model.predict_on_batch([E_X])
					prec = self.precision(y_train[start:end], ypred)
					rec = self.recall(y_train[start:end], ypred)
					fscore = 2*(prec*rec)/(prec+rec)

					print("\tbatch step {}/{}  ".format(end, x_train.shape[0]), end="")
					print("loss: {}, accuracy: {}, f-score: {}, precision: {}, recall: {}".format(loss[0], loss[1], fscore, prec, rec))

					i += self.FLAGS.batch_size

					# This is to free memory
					gc.collect()

				# Calculate the validation accuracy
				val_loss, i = [], 0
				while i < x_val.shape[0]:
					start = i
					end = start + self.FLAGS.batch_size
					batch_size = self.FLAGS.batch_size

					if end > self.FLAGS.num_samples:
					    end = self.FLAGS.num_samples
					    batch_size = end - start

					E_X = self.build_embedding(x_val[start:end])
					E_X = np.array([np.array(sample) for sample in E_X])

					val_loss.append(model.test_on_batch([E_X], y_val[start:end]))

					i += batch_size

				# Saves the last batch's training and validation loss. This is for data plotting
				training_losses.append(loss)
				validation_losses.append(val_loss)

				val_loss = np.average(val_loss, axis=0)
				print("\tval loss: {}, val accuracy: {}".format(val_loss[0], val_loss[1]))

		""" Plot the training and validation learning curve """
		if not os.path.isdir(self.FLAGS.figure_folder):
			os.mkdir(self.FLAGS.figure_folder)

		plt.plot([loss[0] for loss in training_losses])
		plt.plot([loss[0] for loss in validation_losses])
		plt.title('Model loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.savefig(os.path.join(self.FLAGS.figure_folder, "model_learning_curve.png"))

	def eval(self):
		""" This will evaluate the trained model """
		pass
