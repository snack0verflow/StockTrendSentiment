import math
#O(distinct_words)
def computeTF(wordDict):
        tfDict = {}
        wordcount=0
        for key,val in wordDict.items():
        	wordcount+=val
        for key,val in wordDict.items():
                tfDict[key] = val/float(wordcount)
        return tfDict
#O(doc_count*distinct_words)
def computeIDF(docList):
    idfDict = {}
    N = len(docList)
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
            for word, val in doc.items():
                if val>0:
                    idfDict[word] += 1
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))
    return idfDict
#O(distinct_words)
def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word,val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf


'''s1_text="The car is driven on the road"
s2_text="The truck is driven on the highway"
s1_dict={}
s2_dict={}
distinct=[]
s1=s1_text.split(" ")
s2=s2_text.split(" ")
for i in s1:
    if(i not in distinct):
        s1_dict[i]=0
        s2_dict[i]=0
        distinct.append(i)
    s1_dict[i]+=1
for i in s2:
    if(i not in distinct):
        s1_dict[i]=0
        s2_dict[i]=0
        distinct.append(i)
    s2_dict[i]+=1
print("Dictionary for DOC1")
print(s1_dict)

print("Dictionary for DOC2")
print(s2_dict)

print("TF for DOC1")
print(computeTF(s1_dict))

print("TF for DOC2")
print(computeTF(s2_dict))

doc_list=[s1_dict,s2_dict]

print("IDF in General")
print(computeIDF(doc_list))

tf_idf_vector=[]
for i in range(len(distinct)):
    temp=[0,0]
    tf_idf_vector.append(temp)

j=0
for doc in doc_list:
    temp_dict=computeTFIDF(computeTF(doc),computeIDF(doc_list))
    for i in range(len(distinct)):
        tf_idf_vector[i][j]=temp_dict[distinct[i]]
    j+=1

for i in tf_idf_vector:
    print(i)
'''
