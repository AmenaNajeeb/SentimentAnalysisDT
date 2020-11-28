# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:16:50 2020

@author: Amena
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 02:35:43 2020

@author: Amena
"""

from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import pandas as pd
import nltk
nltk.download("punkt")
nltk.download("stopwords")
import re
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler



# load the model from disk
filename = 'nlp_model_dt0.pkl'
dt0 = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform_dt0.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
# -*- coding: utf-8 -*-
# =============================================================================
# """
# Created on Mon Nov 23 00:36:16 2020
# 
# @author: Amena
# """
# 
# import pandas as pd 
# import pickle
# import pandas as pd
# import nltk
# nltk.download("punkt")
# nltk.download("stopwords")
# import re
# from textblob import TextBlob
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from nltk.stem import PorterStemmer
# 
# 
# # =============================================================================
# df = pd.read_csv("C:\\Users\Amena\Downloads\sd_card.csv")
# df.head()
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# stop_words = set(stopwords.words('english'))
# wordnet = WordNetLemmatizer()
# #     
# def change_text(sentence):
#     sentence=str(sentence)
#     sentence=sentence.lower()  #convert into lower case
#     sentence=re.sub(r"http\S+"," ",sentence) #remove hyperlinks
#     cleanr = re.compile('<.*?>')   #remove html tags
#     cleantext = re.sub(cleanr, ' ', sentence)
#     cleantext=re.sub(r'\d+', '', cleantext) #remove numbers
#     cleantext=re.sub(r'[^\w\s]', '', cleantext) #remove punctuations from text
#     cleantext=cleantext.strip()  # Remove leading and trailing '\n'
#     cleantext=" ".join(cleantext.split())  #remove whitespaces
#     word_tokens = nltk.word_tokenize(cleantext)
#     word_tokens=[wordnet.lemmatize(word) for word in word_tokens if word not in stop_words] 
#     cleantext=" ".join(word_tokens)
#     return cleantext
# #     
# df['review_content']=df['review_content'].apply(change_text)
# df['review_content']
# #     
# #     rc = str(df['review_content'])
# #     
# def extract_ratings(ratings):
#     ratings=re.sub('out of 5 stars','',ratings)
#     ratings=int(float(ratings))
#     return ratings
# #     
# df['rating']=df['rating'].apply(extract_ratings)
# df['rating']
# #     
# def sentiment_rating(rating):
# #         # Replacing ratings of 1,2 with -1 (negative), 3 with 0 (neutral) and 4,5 with 1 (positive)
#     if(int(rating) == 1 or int(rating) == 2):
#         return -1
#     elif(int(rating)==3):
#         return 0
#     else: 
#         return 1
# df['sentiment'] = df['rating'].apply(sentiment_rating) 
# #     
# df['polarity'] = df['review_content'].apply(lambda x: TextBlob(x).sentiment.polarity)
# 
# #     
# # =============================================================================
# #      positive = x_train[y_train[y_train == 1].index]
# #      neutral = x_train[y_train[y_train == 0].index]
# #      negative = x_train[y_train[y_train == -1].index]
# # =============================================================================
# #     
# corpus = []
# # =============================================================================
# #     
# for i in range(0, 4520):
#     review = re.sub('[^a-zA-Z]', ' ', df['review_content'][i])
#     review = review.lower()
#     review = review.split()
#     ps = PorterStemmer()
#     review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#     review = ' '.join(review)
#     corpus.append(review)
# #     
# #     
# #     # In[32]:
# #     
# #     
# #     # creating bag of words
# #     
# from sklearn.feature_extraction.text import CountVectorizer
# #     
# cv = CountVectorizer(max_features = 2500)
# #     
# x = cv.fit_transform(corpus).toarray()
# #     
# y = df['sentiment'].values
# #     
# # =============================================================================
# #  print(x.shape)
# #      print(y.shape)
# # =============================================================================
# #     
# pickle.dump(cv, open('tranform_dt0.pkl', 'wb')) 
# 
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 15)
# #     
# # =============================================================================
# #      print(x_train.shape)
# #      print(y_train.shape)
# #      print(x_test.shape)
# #      print(y_test.shape)
# # =============================================================================
# from sklearn.preprocessing import MinMaxScaler
# #     
# mm = MinMaxScaler()
# #     
# x_train = mm.fit_transform(x_train)
# x_test = mm.transform(x_test)
# # 
# # =============================================================================
# dt0 = DecisionTreeClassifier(random_state=0)
# 
# dt0.fit(x_train, y_train)
# 
# y_pred = dt0.predict(x_test)
# 
# print("Training Accuracy :", dt0.score(x_train, y_train))
# print("Testing Accuracy :", dt0.score(x_test, y_test))
# 
# # In[146]:
# y_pred_dt0= dt0.predict(x)  
#      
# filename = 'nlp_model_dt0.pkl'
# pickle.dump(dt0, open(filename, 'wb'))
# # =============================================================================
# y_pred_dt0 = dt0.predict(x)
# =============================================================================

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = dt0.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)