# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import tweepy
import re
import seaborn as sns

import tweepy
import pandas as pd
import numpy as np
drive.mount('drive')
#override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        print(status.text)

consumer_token = 'qK9Id15D378QBaWIeEq4JGIXH'
consumer_secret = 'dasgjb9ERJUVkS2z58FIeiIqP7t2D1iYDaFFeAFhi0XK1T2jQC'
auth = tweepy.OAuthHandler(consumer_token, consumer_secret)

api = tweepy.API(auth,wait_on_rate_limit = True)

df = pd.DataFrame(columns=['Tweet'])

#print(df)
#df.to_csv('data.csv')
#!cp data.csv "drive/My Drive/"

'''
pd.set_option('display.max_colwidth', -1)

# load it into a pandas dataframe
tweet_df = pd.DataFrame(tweet_lst, columns=['tweet_date', 'Keyword', 'id', 'username', 'name', 'tweet', 'like_count', 'reply_count', 'retweet_count', 'retweeted'])
tweet_df.to_csv('tweets.csv')
tweet_df.head()
'''
tweet_df = pd.DataFrame(columns=['tweet', 'like_count', 'reply_count', 'retweet_count', 'retweeted'])
q = "vaccine"
i =1
for tweet in tweepy.Cursor(api.search, q).items(3000):
    tweet_df.loc[i, ['tweet']] = tweet.text
    tweet_df.loc[i, ['retweet_count']] = tweet.retweet_count
    tweet_df.loc[i, ['follower_count']] = tweet.user.followers_count
    i +=1
print(tweet_df)

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt
def clean_tweets(tweets):
    #remove twitter Return handles (RT @xxx:)
    tweets = np.vectorize(remove_pattern)(tweets, "RT @[\w]*:") 
    
    #remove twitter handles (@xxx)
    tweets = np.vectorize(remove_pattern)(tweets, "@[\w]*")
    
    #remove URL links (httpxxx)
    tweets = np.vectorize(remove_pattern)(tweets, "https?://[A-Za-z0-9./]*")
    
    #remove special characters, numbers, punctuations (except for #)
    tweets = np.core.defchararray.replace(tweets, "[^a-zA-Z]", " ")
    
    return tweets

tweet_df['tweet'] = clean_tweets(tweet_df['tweet'])
tweet_df['tweet'].head()

#sentiment analysis with nltk vader
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

sentence = "I like polar bears"
sid.polarity_scores(sentence)

vader_analysis = tweet_df.copy(deep=True)

vader_analysis['neg'] = vader_analysis['tweet'].apply(lambda x:sid.polarity_scores(x)['neg'])
vader_analysis['neu'] = vader_analysis['tweet'].apply(lambda x:sid.polarity_scores(x)['neu'])
vader_analysis['pos'] = vader_analysis['tweet'].apply(lambda x:sid.polarity_scores(x)['pos'])
vader_analysis['compound'] = vader_analysis['tweet'].apply(lambda x:sid.polarity_scores(x)['compound'])
#vader_analysis.drop(['Keyword', 'tweet_date', 'id', 'username', 'name', 'like_count', 'reply_count', 'retweet_count', 'retweeted'], axis=1, inplace=True)
#print(vader_analysis.loc[20])

#Textblob analysis 
from textblob import TextBlob
text_blob_analysis = tweet_df.copy(deep=True) 
#text_blob_analysis.drop(['Keyword', 'tweet_date', 'id', 'username', 'name', 'like_count', 'reply_count', 'retweeted'], axis=1, inplace=True)
def sentiment_blob(x):
  return TextBlob(x)
text_blob_analysis['sentiment'] = text_blob_analysis['tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
text_blob_analysis.head()

TextBlob('this place is the worst').sentiment.polarity

import seaborn as sns
def get_score(row):
  if row['retweet_count'] == 0:
    return 0
  elif row['follower_count'] > 400 and row['follower_count'] < 500:
    return row['retweet_count'] 
  else:
    return 0
text_blob_analysis['score'] = text_blob_analysis.apply(get_score, axis = 1)
vader_analysis['score'] = vader_analysis.apply(get_score, axis = 1 )

sns.lineplot(data= text_blob_analysis, x="sentiment", y="score")
sns.lineplot(data= vader_analysis, x="compound", y="score")

sns.lineplot(data= text_blob_analysis, x="score", y="sentiment")
sns.lineplot(data= vader_analysis, x="score", y="compound")