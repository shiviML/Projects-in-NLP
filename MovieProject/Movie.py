import numpy as np
import re
import pickle 
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import pandas as pd
# Importing the dataset
dataset = pd.read_csv("imdb_master.csv",encoding = "ISO-8859-1")

# Check if any data is null
total = dataset.isnull().sum()

# Only 50000 entries are labelled so taking only labelled data
X = dataset.iloc[:50000,[2]].values
X = [str(x) for x in X]
maps = {'neg' : 0 , 'pos' :1}
dataset['label'] = dataset['label'].map(maps)
y = dataset.iloc[:50000,[3]].values 

#Store this data in file
with open("independent.pickle", "wb") as f:
    pickle.dump(X,f)
with open("dependent.pickle", "wb") as f:
    pickle.dump(y,f)
    
# Preprocessing of data
stop_words = stopwords.words('english') 
corpus = []
for i in range(0, len(X)):
    
    review = re.sub(r'\W', ' ', str(X[i]))
    review = re.sub(r'\d', ' ', review)
    review = review.lower()
    review = re.sub(r'br[\s$]', ' ', review)
    review = re.sub(r'\s+[a-z][\s$]', ' ',review)
    review = re.sub(r'b\s+', '', review)
    review = re.sub(r'\s+', ' ', review)
    # Split the review
    wordlist = nltk.word_tokenize(review)
    temp = []
    for word in wordlist:
        if word not in stop_words:
            temp.append(word)
    new_word = ' '.join(temp)
    corpus.append(new_word)    
    
#Create TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
tiv = TfidfVectorizer(max_features = 8000, min_df = 2, norm="l2", use_idf=True, sublinear_tf = True, max_df = 0.6, stop_words = stop_words)
X = tiv.fit_transform(corpus).toarray()
   
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
text_train, text_test, sent_train, sent_test = train_test_split(X, y, test_size = 0.2)

 # Fitting the Training set to Linear SVC
from sklearn.svm import LinearSVC
classifier = LinearSVC(C = 0.1)
classifier.fit(text_train,sent_train)   
    
 # Pickling classifier
with open('svcclassifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    

# Pikling TF-IDF model
with open('TFIDF.pickle','wb') as f:
    pickle.dump(tiv,f)    

#Unpickling 
with open("svcclassifier.pickle" ,'rb') as f:
    classifier = pickle.load(f)  

# Predicting the Test set results
sent_pred = classifier.predict(text_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, roc_curve
cm = confusion_matrix(sent_test, sent_pred)

##Computing false and true positive rates
fpr, tpr,_=roc_curve(sent_pred,sent_test,drop_intermediate=False)

import matplotlib.pyplot as plt
plt.figure()
##Adding the ROC
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')

plt.savefig('ROC.jpg')
plt.show()
    
    
    
    