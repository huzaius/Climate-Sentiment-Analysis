"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""

# Core Pkgs
from matplotlib.ft2font import HORIZONTAL
from nbformat import write
import streamlit as st 
import altair as alt
import plotly.express as px 
from streamlit_option_menu import option_menu
import matplotlib as plt



# NLP
import nltk
from collections import Counter
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import string
import re
stop = stopwords.words('english')
# from wordcloud import WordCloud, STOPWORDS
from PIL import Image 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

# Utils
import pickle 
lsvc_model = "resources/lsvc_pipe.pkl"
pipe_lr = pickle.load(open(lsvc_model,"rb"))

# Part of Speech for modeling
def POS(word):
	"""
	This function gets the part of speech
	"""
	pos_counts = Counter()
	probable_part_of_speech = wordnet.synsets(word)
	pos_counts["n"] = len([i for i in probable_part_of_speech if i.pos()=="n"])
	pos_counts["v"] = len([i for i in probable_part_of_speech if i.pos()=="v"])
	pos_counts["a"] = len([i for i in probable_part_of_speech if i.pos()=="a"])
	pos_counts["r"] = len([i for i in probable_part_of_speech if i.pos()=="r"])
	part_of_speech = pos_counts.most_common(1)[0][0]
	return part_of_speech

# remove the urls, hashtags, mentions
def cleaner1(text):
	
	new_text = re.sub(r'''#\w+ ?''', '', text) # remove hashtags
	new_text = re.sub(r'''http\S+''', '', new_text)# Remove URLs
	new_text = re.sub(r"(?:\@|https?\://)\S+", "", new_text)# Remove mentions
	new_text = re.sub(r'[^\w\s]', '', new_text) # remove punctuations
	return new_text


def cleaner2(new_text):
	'''
	This function cleans the tweets by tokenizing, removing punctuation, 
	removing digits and removing 1 character tokens
	
	'''

	# tokenizing the tweets
	clean_text = new_text.split() ## first we tokenize

	# removing digits from the tweets
	clean_tweets = [token for token in clean_text if token not in list(string.digits)]

	# lastly we remove all one character tokens
	clean_text = [token for token in clean_text if len(token) > 1]
	# Convert to lower case
	clean_text = (map(lambda x: x.lower(), clean_text))
	clean_text = [item for item in clean_text if item not in stop]  # Remove stopwords
	clean_text = [WordNetLemmatizer().lemmatize(token, POS(token)) for token in clean_text]

	return clean_text

def predict_emotions(clean_text):
	results = pipe_lr.predict([clean_text])
	if results == -1:
		return "Anti sentiment"
	elif results == 0:
		return "Neutral sentiment"
	elif results == 1:
		return "Pro sentiment"
	else:
		return "News Sentiment"


# def wordcount_gen(df, sent_labels):
#     '''
#     Generating Word Cloud
#     inputs:
#        - df: tweets dataset
#        - category: Positive/Negative/Neutral
#     '''
#     # Combine all tweets
#     combined_tweets = " ".join([tweet for tweet in df[df.sent_labels==sent_labels]['Cleaned_Tweet'].str.join(" ")])
                          
#     # Initialize wordcloud object
#     wc = WordCloud(background_color='grey', 
#                    max_words=50, 
#                    stopwords = STOPWORDS)

#     # Generate and plot wordcloud
#     plt.figure(figsize=(10,10))
#     plt.imshow(wc.generate(combined_tweets))
#     plt.title('Top 50 {} Sentiment Words'.format(sent_labels), fontsize=20)
#     plt.axis('off')
#     plt.show()

# # Main Application
def main():
	st.title("Climate Change Sentiment Analyzer")

	with st.sidebar:
		selected = option_menu("Main Menu", ["Home","EDA", "Model", "About"],
            icons=['house', 'bar-chart-fill', 'kanban','person lines fill'], menu_icon="cast")
    

	if selected == "Model":
		#st.subheader("Machine Learning Classifier")

		dropdown = st.sidebar.selectbox (" Select a Model ", ["Linear SVC","Logistic Regression", "Ridge Regression"])
		if dropdown == "Linear SVC":
			st.subheader("Linear SVC Machine Learning Classifier")
			
			#Linear SVC Model Prediction
			with st.form(key='sentiment_form'):
				raw_text = st.text_area("Type Here or copy and paste tweet")
				submit_text = st.form_submit_button(label='Submit to predict')

			if submit_text:
				col1,col2  = st.columns(2)

	# 			# Apply Fxn Here
				text = cleaner1(raw_text)
				clean_text = cleaner2(text)
				prediction = predict_emotions(' '.join(clean_text))
				with col1:
					st.success("Original Text")
					st.write(raw_text)
					st.write("Text length: {}".format(len(raw_text.split())))

				st.success('Prediction Using Linear SVC')
				st.write("The text or tweet is indicative of a {}".format(prediction))



				with col2:
					st.success("Text used in prediction")
					st.write(' '.join(clean_text))
					st.write("Text Lenght: {}".format(len(clean_text)))

		elif dropdown == "Logistic Regression":
			st.subheader("Logistic Regression Machine Learning Classifier")

		#Logistic Regression Model Prediction
			with st.form(key='sentiment_form'):
				raw_text = st.text_area("Type Here or copy and paste tweet")
				submit_text = st.form_submit_button(label='Submit to predict')

			if submit_text:
				col1,col2  = st.columns(2)

	# 			# Apply Fxn Here
				text = cleaner1(raw_text)
				clean_text = cleaner2(text)
				prediction = predict_emotions(' '.join(clean_text))
				with col1:
					st.success("Original Text")
					st.write(raw_text)
					st.write("Text length: {}".format(len(raw_text.split())))

				st.success('Prediction Using Logistic Regression')
				st.write("The text or tweet is indicative of a {}".format(prediction))



				with col2:
					st.success("Text used in prediction")
					st.write(' '.join(clean_text))
					st.write("Text Lenght: {}".format(len(clean_text)))
		
		#Ridge Regression
		elif dropdown == "Ridge Regression":
			st.subheader("Ridge Regression Machine Learning Classifier")

			with st.form(key='sentiment_form'):
				raw_text = st.text_area("Type Here or copy and paste tweet")
				submit_text = st.form_submit_button(label='Submit to predict')

			if submit_text:
				col1,col2  = st.columns(2)

	# 			# Apply Fxn Here
				text = cleaner1(raw_text)
				clean_text = cleaner2(text)
				prediction = predict_emotions(' '.join(clean_text))
				with col1:
					st.success("Original Text")
					st.write(raw_text)
					st.write("Text length: {}".format(len(raw_text.split())))

				st.success('Prediction Using Ridge Regression')
				st.write("The text or tweet is indicative of a {}".format(prediction))



				with col2:
					st.success("Text used in prediction")
					st.write(' '.join(clean_text))
					st.write("Text Lenght: {}".format(len(clean_text)))

		
	
	elif selected == "EDA":
		st.subheader("Exploratory Data Analysis")
		df = pd.read_csv("Clean_text_data.csv")
		# st.dataframe(df)

		st.success("Distribution of sentiments")
		fig = px.pie(df, names='sent_labels')
		st.plotly_chart(fig, use_container_width=True)

		st.success("Distribution of Tweet Lenght By Sentiment")
		fig1 = px.box(df, x='sent_labels', y='Original_Tweet_length')
		st.plotly_chart(fig1, use_container_width=True)

		col1, col2 = st.columns(2)
		with col1:
			st.success("Original Tweet length Dist")
			hist1 = px.histogram(df, x='Original_Tweet_length', histnorm='density', marginal='box',)
			st.plotly_chart(hist1, use_container_width=True)
		with col2:
			st.success("Cleaned Tweet length Dist")
			hist2 = px.histogram(df, x='Cleaned_Tweet_length', histnorm='density', marginal='box', color_discrete_sequence=['green'])
			st.plotly_chart(hist2, use_container_width=True)

		st.success('Click the buttons below to display their respective wordclouds')
		button1, button2, button3, button4 = st.columns([1,1,1, 1])

		with button1:
			anti = st.button('Anti Sentiment')
		with button2:
			newtral = st.button('Neutral Sentiment')
		with button3:
			pro = st.button('Pro Sentiment')
		with button4:
			news = st.button('News Sentiment')

		anti_img = Image.open('resources/imgs/anti.JPG')
		neutral_img = Image.open('resources/imgs/neutral.JPG')
		pro_img = Image.open('resources/imgs/pro.JPG')
		news_img = Image.open('resources/imgs/news.JPG')

		
		if anti:
			st.image(anti_img, caption='')
		elif newtral:
			st.image(neutral_img, caption='')
		elif pro:
			st.image(pro_img, caption='')
		elif news:
			st.image(news_img, caption='')

	elif selected == "Home":
		climate_img = Image.open('resources/imgs/climate_.jpg')
		sentiment_img = Image.open("resources/imgs/sentiment.jpg")
		st.subheader("What is Climate Change?")
		st.write('Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels (like coal, oil and gas), which produces heat-trapping gases.')
		st.image(climate_img)

		st.subheader("What is Sentiment Analysis?")
		st.write("Sentiment analysis is the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information.")
		st.image(sentiment_img)

	elif selected == "About":

		st.write("This is a climate change sentiment analyzer that uses three(3) different models to analyse, tweets or text and categorises it into sentiments. This text analyzed wit be categorized into News,Pro,Neutral and Negetive sentiments.")
		st.write("The Home region explains what sentiment analysis and climate change is whiles the EDA section shows the distribution of the cleaned dataset. The model section is where the text are categorised using Linear SVC, Logistic Regression and Ridge Regression models.")
		


		
if __name__ == '__main__':
	main()