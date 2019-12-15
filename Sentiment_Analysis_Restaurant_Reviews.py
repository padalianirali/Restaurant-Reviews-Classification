'''
Step 1: Importing all required libraries    
'''
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

'''
Step 2: Importing restaurant reviews dataset    
'''
path = "E:\\Training\\NLP\\SA_Restaurant_Reviews\\Data_Restaurant_Reviews.tsv"
data = pd.read_csv(path, delimiter="\t")
print(data.head())
#path shows the source of data
#delimiter defines the way is which data is seperated in text files - in this case it is a tsv file (tab seperated file) so we need to specify \t

'''
Step 3: Performing data preprocessing
'''
corpus=[]
#defining loop till 1000 as our data contains 1000 reviews in total
for i in range (0,1000):
    #replaces all characters except a-z and A-Z with one space for cleaning
    review = re.sub("[^a-zA-Z]"," ",data["Review"][i])
    
    #replaces all uppercase characters in a review with lowercase characters
    review = review.lower()
    
    #creates a list of comma seperated words for each review
    review = review.split()
    
    #removes all stopwords as per english dictionary
    ps_obj = PorterStemmer()
    review = [ps_obj.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    #combines comma seperated words for each review within a list with space
    review = " ".join(review)
    
    #appends all reviews in a single list
    corpus.append(review)

'''
Step 4: Preparing independent (x) and dependent (y) variables for the model
'''
from sklearn.feature_extraction.text import CountVectorizer
#creating a count vector object for top 1500 most frequent words to be used as features
cv = CountVectorizer(max_features=1500)
#creating (fitting and transforming) a count vector of the corpus using fit_transform() method
#converting the encoded vector into an array using toarray() method
x = cv.fit_transform(corpus).toarray()
#select values from all rows of the second column of a dataframe
y = data.iloc[:,1].values

'''
Step 5: Splitting data into training (80%) and testing (20%) sets
'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)
#random_state helps to retain the same dataset for training and testing everytime the code is executed

'''
Step 6: Training the model
'''
from sklearn.naive_bayes import GaussianNB
NB_model = GaussianNB()
NB_model.fit(x_train, y_train)

'''
Step 7: Testing the model
'''
y_predict = NB_model.predict(x_test)

'''
Step 8: Evaluating the model
'''
from sklearn.metrics import confusion_matrix
NB_model_cm = confusion_matrix(y_test, y_predict)
print(NB_model_cm)







   
    





