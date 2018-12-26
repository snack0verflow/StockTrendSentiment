from nltk.stem.snowball import SnowballStemmer
import pickle
import numpy as np

stemmer=SnowballStemmer('english')

#Load the distinct words list
with open('distinct.pkl','rb') as f:
	distinct_words=pickle.load(f)

#Load the TF_IDF vector
with open('tf_idf.pkl','rb') as f:
	TF_IDF_VECTOR=pickle.load(f)

print("Please enter the statement/tweet")
print()
testing=raw_input()
special_characters=['[',']','\\','/',',','"','@','#','.']

for i in special_characters:
	testing=testing.replace(i,"")

testing=testing.split(" ")
testing1=[]

for i in range(len(testing)):
	testing1.append(stemmer.stem(testing[i]))
	#print(testing1[i])

testing_row=[]
for i in range(len(distinct_words)):
	if(distinct_words[i] in testing1):
		testing_row.append(1)
	else:
		testing_row.append(0)

ans=np.matmul(testing_row,TF_IDF_VECTOR)
emotions=['Joy','Fear','Anger','Sadness','Disgust','Shame','Guilt']
mx=max(ans)
#print(ans)
#print(mx)
if(mx==0):
	print("No results found")
else:
	for i in range(len(ans)):
		ans[i]=ans[i]/(float)(mx)

	results=[]
	for i in range(len(ans)):
		results.append((ans[i],emotions[i]))
	results.sort()
	results=results[::-1]
	for i in results[:3]:
		print(i[1],i[0])



