#This will read the CSV file
#Makes documents according to the sentiments
#happy.txt, sad.txt 
#tokenization and lemmatization done on each of them
#tf-idf indexing done on each of them

#now a day's(assume 100) tweets will be stored and for each tweet we will score it
#on the basis of it's sentiment score

#each sentiment's score for a day will be stored

'''
Choice of stop words is very subjective
We will use the NLTK package's stop words list to remove stop words from out documents. TF IDF filters aren't good enough here.
Minimum term frequency removal can actually prove counter-productive here.
We will remove some words from the stop words list which seem to be contributing to the general sentiment
'''

'''
All these requirements lead us to  use ISEAR (International Survey  on  Emotion  Antecedents  and  Reactions)  dataset  in  our  
experiments. ISEAR consists of 7,666 sentences and snippets in which 1096 participants from fields of psychology, social   
sciences, languages, fine arts, law, natural sciences, engineering and medical in 16 countries across five continents completed a  
questionnaire about the experiences and reactions to seven emotions in everyday  life including joy, fear, anger, sadness,  
disgust, shame and guilt.
'''

import nltk
from nltk.corpus import stopwords
import csv
from nltk.stem.snowball import SnowballStemmer
import pickle
import math
import numpy as np
import re
from tf_idf import *

# This ia a function to remove non-ascii characters from input
def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i)<128)

# Preprocessing of a tweet
def processTweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip('\'"')
    return tweet

#Some words present in the original stopwords may be beneficial to us, so retain them
remove_from_stop=['not','are','can','will','no','nor','very','again','with','about','against','between','through','during'
'before','after','above','below','further','all','few','most','more','out','have','has','had','having']

#Usage list of stopwords
stop_list=[i for i in stopwords.words('english') if i[-3:]!="n't" and i not in remove_from_stop]

#Dictionaries for each emotion
joy = {}
fear = {}
anger = {}
sadness = {}
disgust = {}
shame = {}
guilt = {}

#Initializing NLTK's SnowballStemmer object
stemmer = SnowballStemmer("english")
#Reading the ISEAR dataset
reader=csv.reader(open('isear.csv','r'))
count=0
#Special characters that need to be stripped off 
special_characters=['[',']','\\','/',',','"','@','#','.']
distinct_words=[]							#A list of distinct words over the corpora
for row in reader:
	l=row[0].split('|')						#Split the row
	emotion=l[36]							#Emotion for the text														
	text=l[40]								#Text of the document																			
	text=remove_non_ascii(text)				#Remove non ASCII													
	for i in special_characters:			#Replacing special characters with blank															
		text=text.replace(i,"")
	tokens=text.split(" ")

	for i in tokens:						#stem the tokens
		if stemmer.stem(i) not in distinct_words:												
			distinct_words.append(stemmer.stem(i))
			joy[stemmer.stem(i)]=0
			fear[stemmer.stem(i)]=0
			anger[stemmer.stem(i)]=0
			sadness[stemmer.stem(i)]=0
			disgust[stemmer.stem(i)]=0
			shame[stemmer.stem(i)]=0
			guilt[stemmer.stem(i)]=0

		#print(text)
		
		if(emotion=='joy'):					#Frequency updation of stemmed words for each emotion document
			joy[stemmer.stem(i)]+=1
		elif(emotion=='fear'):		
			fear[stemmer.stem(i)]+=1
		elif(emotion=='anger'):		
			anger[stemmer.stem(i)]+=1
		elif(emotion=='sadness'):		
			sadness[stemmer.stem(i)]+=1
		elif(emotion=='disgust'):		
			disgust[stemmer.stem(i)]+=1
		elif(emotion=='shame'):		
			shame[stemmer.stem(i)]+=1
		elif(emotion=='guilt'):		
			guilt[stemmer.stem(i)]+=1

print("Dictionaries prepared")
print(len(distinct_words))

'''
with open('joy.pkl','wb') as f:
	pickle.dump(joy,f)
with open('fear.pkl','wb') as f:
	pickle.dump(fear,f)
with open('anger.pkl','wb') as f:
	pickle.dump(anger,f)
with open('sadness.pkl','wb') as f:
	pickle.dump(sadness,f)
with open('disgust.pkl','wb') as f:
	pickle.dump(disgust,f)
with open('shame.pkl','wb') as f:
	pickle.dump(shame,f)
with open('guilt.pkl','wb') as f:
	pickle.dump(guilt,f)
with open('distinct.pkl','wb') as f:
	pickle.dump(distinct_words,f)
'''
doc_list=[joy,fear,anger,sadness,disgust,shame,guilt]
print("Prepared Document List")

TF_IDF_VECTOR=[]
IDF=computeIDF(doc_list)				#Calculate the IDF from the imported module
print("IDF calculated")

#2-D array initialisation
for word_count in range(len(distinct_words)):
	temp=[0 for i in range(7)]
	TF_IDF_VECTOR.append(temp)
print("2-D Array initialized")

#TF-IDF matrix being filled
for doc_iter in range(len(doc_list)):
	temp_dict=computeTFIDF(computeTF(doc_list[doc_iter]),IDF)
	for word_count in range(len(distinct_words)):
		TF_IDF_VECTOR[word_count][doc_iter]=temp_dict[distinct_words[word_count]]
	print(doc_iter)


with open('tf_idf.pkl','wb') as f:
	pickle.dump(TF_IDF_VECTOR,f)

#Testing snippet


