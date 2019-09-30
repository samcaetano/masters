from pytorch_pretrained_bert import BertTokenizer
import os
import json

class Utils():
	def __init__(self, dataset, FLAGS, bert_model='bert-base-uncased'):
		self.CLS = "[CLS] "
		self.SEP = " [SEP]"
		self.word2idx = {"[PAD]": 0}
		self.bert_pretrained_model = bert_model
		self.dataset = dataset
		self.FLAGS = FLAGS
		self.preprocessed_filename = "preprocessed_"+self.bert_pretrained_model+"_"+self.dataset+".json"
		self.vocab_filename = "vocabulary_"+self.bert_pretrained_model+"_"+self.dataset+".json"

		self.tokenizer = BertTokenizer.from_pretrained(self.bert_pretrained_model)

	def load(self, path):
		if self.dataset == "byarticles":
			with open(os.path.join(path, "articles-byarticles-training.json"), "r") as f:
				X = json.load(f)
			with open(os.path.join(path, "groundtruth-byarticles-training.json"), "r") as f:
				Y = json.load(f)

			return X, Y

		elif self.dataset == "bypublisher":
			print("Not implemented yet")

	def load_preprocessed(self):
		file_path = os.path.join(self.FLAGS.checkpoint_folder, self.preprocessed_filename)
		
		if os.path.exists(file_path):
			with open(file_path, "r") as f:
				X = json.load(f)
		
			print("Preprocessed loaded")
			return X
		else:
			print("Need to preprocess the data first.")

	def pad_sequences(self, X):
		for sample in X:
			sample += [0 for _ in range(self.FLAGS.BERT_MAX_LEN-len(sample))]


	def preprocessing(self, texts):
		# Add special tokens [CLS] and [SEP] to sentences
		for i, text in enumerate(texts):
			sentences = text[0].split(". ") # split by sentences
			text = ' [SEP] '.join(sentences)
			text = self.CLS + text + self.SEP
			texts[i] = text

		for i, text in enumerate(texts):
			# Tokenize sentences
			tokenized_text = self.tokenizer.tokenize(text)

			# Clip to the maximum BERT's model sequence's length
			tokenized_text = tokenized_text[:self.FLAGS.BERT_MAX_LEN]

			# Find the BERT's index for each token in the sentence
			indexed_text = self.tokenizer.convert_tokens_to_ids(tokenized_text)

			# Build word to index map
			for word, idx in zip(tokenized_text, indexed_text):
				self.word2idx.update({word:idx})

			texts[i] = indexed_text

		# Padding to the max length
		self.pad_sequences(texts)

	def save_preprocessed(self, X): 
		""" Saves the embedding to a file """

		if not os.path.isdir(self.FLAGS.checkpoint_folder):
			os.mkdir(self.FLAGS.checkpoint_folder)

		with open(os.path.join(self.FLAGS.checkpoint_folder, self.preprocessed_filename), "w") as f:
			json.dump(X, f)
			print("Preprocessed saved")

		with open(os.path.join(self.FLAGS.checkpoint_folder, self.vocab_filename), "w") as f:
			json.dump(self.word2idx, f)
			print("Vocabulary saved")
