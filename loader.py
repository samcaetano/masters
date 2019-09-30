"""
 File for  loading the given dataset "byarticles" or "bypublisher".
"""
import argparse
import re
from bs4 import BeautifulSoup
import json

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Choose 'byarticles' or 'bypublisher' (no quotes)")

arg = parser.parse_args()

class Loader():
	html_regex = '<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});'

	def load_byarticles(self):
		articles = open("dataset/articles-training-byarticle-20181122.xml", "r")
		groundtruth = open("dataset/ground-truth-training-byarticle-20181122.xml", "r")

		return articles, groundtruth

	def load_bypublisher(self):
		articles_training = open("dataset/articles-training-bypublisher-20181122.xml", "r")
		groundtruth_training = open("dataset/ground-truth-training-bypublisher-20181122.xml", "r")
		articles_val = open("dataset/articles-validation-bypublisher-20181122.xml", "r")
		groundtruth_val = open("dataset/ground-truth-validation-bypublisher-20181122.xml", "r")

		return articles_training, groundtruth_training, articles_val, groundtruth_val

	def preprocessing(self, X, Y):

		X , Y= X.read(), Y.read()
		X, Y = X.split("<article "), Y.split("<article ")

		dataX, dataY = [], []
		for article, label in zip(X[1:], Y[1:]):
			# replace '\n', '\r' and '\t' to '.'
			article = article.replace("\n", ". ")
			article = article.replace("\r", ". ")
			article = article.replace("\t", ". ")

			# get text (title + content)
			article = re.findall(r"title=.*</article>", article, 0)

			# split text by paragraph
			article = article[0].split("<p>")
			article = ' '.join(article)
			
			# remove HTML tags from article text			
			article = BeautifulSoup(article, "lxml").text

			# replace 'title="', '">' and '\'
			article = article.replace("title=\"", "")
			article = article.replace("\">", "")
			article = article.replace("\'", "'")
			article = article.replace(" _", " ")
			article = article.replace("–", "-")
			article = article.replace("”", "\"")
			article = article.replace("“", "\"")
			article = article.replace("’", "'")

			# Get hyperpartisan label
			label = re.findall("hyperpartisan=\"\w+\"", label)

			if "true" in label[0]:
				dataY.append(1) # it is hyperpartisan
			elif "false" in label[0]:
				dataY.append(0) # it isn't

			dataX.append([article])

		return dataX, dataY

loader = Loader()
if arg.dataset == "byarticles":
	X, Y = loader.load_byarticles()

	X, Y = loader.preprocessing(X, Y)

	with open("processed_data/articles-byarticles-training.json", "w") as f:
		json.dump(X, f)

	with open("processed_data/groundtruth-byarticles-training.json", "w") as f:
		json.dump(Y, f)

	print("OK. Loading byarticles done. Processed data saved in folder processed_data/")

elif arg.dataset == "bypublisher":
	print("Not implemented yet")