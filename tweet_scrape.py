import nltk
from nltk.corpus import stopwords
import csv
from nltk.stem.snowball import SnowballStemmer
import pickle
import math
import numpy as np
import re
import os
import xlrd
with open ('distinct.pkl','rb') as f:
	distinct_words=pickle.load(f)
print(len(distinct_words))

special_characters=['[',']','\\','/',',','"','@','#','.']



def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i)<128)

def processTweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip('\'"')
    return tweet

remove_from_stop=['not','are','can','will','no','nor','very','again','with','about','against','between','through','during'
'before','after','above','below','further','all','few','most','more','out','have','has','had','having']

stop_list=[i for i in stopwords.words('english') if i[-3:]!="n't" and i not in remove_from_stop]

stemmer = SnowballStemmer("english")

fl=os.listdir('.')#list of files in the current directory
c=0
#iterating through the files list fl
for j in fl:
	tweets=[]
	try:
		wb=xlrd.open_workbook(j)
	except:#if any non xlsx files are present in the current directory
		continue
	
	sheet=wb.sheet_by_index(1)
	n=sheet.nrows
	#iterating through rows in the datasheet
	for i in range(1,n):
		twt=sheet.cell(i,6)
		date=sheet.cell(i,1)
		#converting from unicode to ascii
		twt=twt.value.encode('ascii','ignore')
		date=date.value.encode('ascii','ignore')
		twt=processTweet(twt)
		for k in special_characters:
			twt=twt.replace(k,"")

		twt=twt.split(" ")
		twt1=[]

		for k in range(len(twt)):
			twt1.append(stemmer.stem(twt[k]))
			twt1[k]=twt1[k].encode('ascii','ignore')

		twt1.append(date)
		tweets.append(twt1)
	#dumping the pickle
	with open (j[17:]+'.pkl','wb') as f:
		pickle.dump(tweets,f)

