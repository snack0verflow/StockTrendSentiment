import sklearn
from sklearn import linear_model
import numpy as np
import pickle 
import nltk
from nltk.corpus import stopwords
import csv
from nltk.stem.snowball import SnowballStemmer
import pickle
import math
import numpy as np
import re
from tf_idf import *
import matplotlib.pyplot as plt

#Load the distinct words list
with open('distinct.pkl','rb') as f:
	distinct_words=pickle.load(f)

#Load the American Airlines tweets
with open('aal.pkl','rb') as f:
	tweets_temp=pickle.load(f)

#Load the TF_IDF vector
with open('tf_idf.pkl','rb') as f:
	TF_IDF_VECTOR=pickle.load(f)

#NTK's SnowballStemmer
stemmer=SnowballStemmer("english")

#Special characters to be stripped
special_characters=['[',']','\\','/',',','"','@','#','.']

day_to_index={}	#dictionary to map dates in string to index
count=0		
tot=92			#Since total number of days was 92
for tw in tweets_temp:
	if(tw[-1] not in day_to_index):
		day_to_index[tw[-1]]=tot-count-1
		count+=1
	#print(tw[-1])
print("Distinct days ", len(day_to_index))
print("Tweet count ",len(tweets_temp))
'''
for key,val in day_to_index.items():
	print(key,val)
'''
#data will contain the emotion values for each day
data=[]
for i in range(92):
	temp=[0,0,0,0,0,0,0]
	data.append(temp)

tweet_count=1

#Process each tweet
for tw1 in tweets_temp:
	if(tweet_count%100==1):
		print("Tweets processed ",tweet_count,"/",len(tweets_temp))
	tweet_count+=1

	date=day_to_index[tw1[-1]]		#date of the in-process tweet
	tw1=tw1[0:-1]					#remove the date token
	testing1=[]						#will contain the stemmed tokens
	tw=[]							#remove special character tokens

	#special characters removal
	for i in tw1:
		if(i not in special_characters):
			tw.append(i)

	#stemming each word
	for i in range(len(tw)):
		testing1.append(stemmer.stem(tw[i]))

	#convert into 1*6072 vector
	testing_row=[]
	for i in range(len(distinct_words)):
		if(distinct_words[i] in testing1):
			testing_row.append(1)
		else:
			testing_row.append(0)

	ans=np.matmul(testing_row,TF_IDF_VECTOR)
	#now ans contains the score for each emotion
	#add the values to respective slots for the day on which the tweet was made
	for i in range(len(ans)):
		data[date][i]+=ans[i]

#normalize each day's emotion scores
for row in data:
	mx=sum(row)
	for j in range(len(row)):
		row[j]=float(row[j])/mx

#Make different lists for visualization purposes
close_data=[]
open_data=[]
joy=[]
fear=[]
anger=[]
sadness=[]
disgust=[]
shame=[]
guilt=[]

for i in range(len(day_to_index)):
	close_data.append(0)
	open_data.append(0)



reader=csv.reader(open('AAL.csv','r'))
no_tweets=[]
for row in data:
	joy.append(row[0])
	fear.append(row[1])
	anger.append(row[2])
	sadness.append(row[3])
	disgust.append(row[4])
	shame.append(row[5])
	guilt.append(row[6])
	

for row in reader:
	#print(row)
	if(row[0] not in day_to_index):
		no_tweets.append(row[0])
	if(row[0]!='Date' and row[0] in day_to_index):
		date=day_to_index[row[0]]
		open_data[date]=float(row[1])
		close_data[date]=float(row[4])

x_axis=[i for i in range(len(day_to_index))]									 	#day count for the visualisation graphs
#############################################
#Filling in values for absent financial data on weekends
i=0
j=0
while(i<len(open_data)):
	j=i+1
	while(j<len(open_data) and open_data[j]==0):
		open_data[j]=open_data[i]
		j+=1
	j=i+1

	while(j<len(open_data) and close_data[j]==0):
		close_data[j]=close_data[i]
		j+=1

	i=j
##############################################



fig,(ax1,ax2)=plt.subplots(2,1)
ax1.plot(x_axis,open_data,label='opening')
ax1.plot(x_axis,close_data,label='closed')
ax1.legend(loc="upper right")

ax2.plot(x_axis,joy,label='happiness')
ax2.plot(x_axis,fear,label='fear')
ax2.plot(x_axis,anger,label='anger')
ax2.plot(x_axis,sadness,label='sadness')
#ax2.plot(x_axis,disgust,label='disgust')
#ax2.plot(x_axis,shame,label='shame')
#ax2.plot(x_axis,guilt,label='guilt')
ax2.legend(loc="upper right")
plt.show()

#print(no_tweets)
