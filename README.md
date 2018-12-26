ISEAR
The International Survey on Emotion Antecedents and Reactions as a python dataset for MachineLearning

Purpose
The purpose of this code repository is to provide a python loader for the Isear Data set
The ISEAR dataset has been developped by the Swiss National Center of Competence in Research.

Basic documentation
The Isear.csv file
This file is the actual extract of the dataset, which was provided as an Access database.
The data has been cleaned and normalized a bit 

Simple working
A basic vector-space-model based sentiment analysis system. 

We simply treat different emotions as different documents in the corpus. The best result according to the ranked-retrieval
will be most likely the emotion for the query. 

Since we've already preprocessed and put in the pkl files which contains the TF-IDF and the distinct words, there's no need to run
the pro_main.py and tf_idf.py again. We can just run the interact.py file and put in the string as an input whose sentiment will be analysed.

Team 
1) Ansuman Dibyayoti Mohanty - 2016A7PS0043H 
2) Syed Abid Abdullah - 2016A7PS0562H 
3) Deepak Gupta - 2016A7PS0105H 
4) Mridul Bhaskar - 2016A7PS0391H 
