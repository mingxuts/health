'''

Created on Jun 21, 2018

@author: longgu

'''

 

import pandas as pd

#from pandas_ml import ConfusionMatrix

from sklearn.metrics import accuracy_score,confusion_matrix

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder

 

import nltk

from sklearn.tree import DecisionTreeClassifier

 

MAX_NUM_WORDS = 10000

 

# This function needs to revised

def data_processing():

    #Loading Data

    data = pd.read_csv('../data/textmining-dos.csv',encoding = "ISO-8859-1")

   

    #Drop unnecessary columns

    dropColumns = ['Random Seed', 'Random Digit', 'Find Random 1\'s', 'DIR Number',

           'Licence ID', ' MR ', ' MRS ', ' MISS ',

           ' MS ', ' DR. ', ' DR ', 'Name Hit']

    #run droping columns

    data_df = data.drop(dropColumns, axis=1)

    

    #remove data whose Investigated is not in ['No',"Yes"]

    data_df = data_df[data_df['Actual Harm'].isin(['Serious Injury', 'Temporary Injury', 'Death', 'No Injury', 'Not Known'])]

   

    #remove data whose description is -

    data_df = data_df[data_df['Actual Harm']!='-']

    

    #Transform Yes and No to 1 and 0

    #data_df.loc[data_df.Investigated =='No','Investigated'] = 0

    #data_df.loc[data_df.Investigated =='Yes','Investigated'] = 1

   

    return data_df

 

def read_glove_vecs(glove_file):

    with open(glove_file, 'r') as f:

        words = set()

        word_to_vec_map = {}

       

        for line in f:

            line = line.strip().split()

            curr_word = line[0]

            words.add(curr_word)

            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

           

    return words, word_to_vec_map

 

def sentence_avg(sentence,words_vecmap,dictionary,stop_words):

    # A list to aggreat

    sen_sum = []

    words = filter(lambda x: not x in stop_words and x.isalpha(), sentence)

    for word in words:

        if word in dictionary:

            sen_sum.append(words_vecmap[word])

    vec = np.array(sen_sum).mean(0).tolist()

    return vec

 

 

def model_embedding(dataframe,words_vecmap,dictionary,stop_words):

   

    tokenize = lambda doc: nltk.word_tokenize(doc.lower())

    X = dataframe['Published Event Description (DAEN)'].apply(tokenize)

   

    ls = []

   

    for i in X:

        ls.append(sentence_avg(i,words_vecmap,dictionary,stop_words))

   

    return np.array(ls)


def model_tfidf(dataframe):

  text = dataframe['Published Event Description (DAEN)'].values

  vectorizer = TfidfVectorizer(ngram_range=(1,1))

  vectorizer.fit(text)

  print(vectorizer.vocabulary_)

  print(vectorizer.idf_)

  # encode document

  vector = vectorizer.transform(text)

  # summarize encoded vector

  print(vector.shape)

  return vector.toarray()

 

  

 

if __name__ == "__main__":

   

    nltk.download('punkt')

    nltk.download('stopwords')

    stop_words = nltk.corpus.stopwords.words('english')

   

    print('downloading nltk')

   

    data_df = data_processing()

    le = LabelEncoder()

    data_df['Actual Harm'] = le.fit_transform(data_df['Actual Harm'])   

    

    data_df_test = data_df.head(10)

    

#     words, words_vecmap = read_glove_vecs('../data/glove.6B.50d.txt')

#     dictionary = list(words_vecmap.keys())

    #print("loading word vec")

    #X = model_embedding(data_df_test,words_vecmap,dictionary,stop_words)

 

    #X = model_tfidf(data_df_test)

    #print( X )

    #print(X.shape)

    X = [x for x in data_df_test['Published Event Description (DAEN)'].values]

    #y = data_df_test['Actual Harm'].astype(int)

    y = data_df_test['Actual Harm']

    print(y)

 

    print(data_df.shape)

    

    C = 1.0


    rdf_clf = RandomForestClassifier()

    dec_clf = DecisionTreeClassifier()

    log_clf = LogisticRegression(C=C, multi_class='multinomial', solver='newton-cg')

 

    pip_rdf = Pipeline([

            ("tfidf", TfidfVectorizer(ngram_range=(1,1))),

            ("random_forest", rdf_clf)

        ])

 

    pip_log = Pipeline([

            ("tfidf", TfidfVectorizer(ngram_range=(1,1))),

            ("logistic", log_clf)

        ])

   

    pip_dec = Pipeline([

            ('tfidf', TfidfVectorizer(ngram_range=(1,1))),

            ('decision_tree', dec_clf)

        ])

 

    all_models = [

        ("random_forest", pip_rdf),

        ("decision tree", pip_dec),

        ("logistic", pip_log),

    ]

 

    for name, model in all_models:

        model.fit(X, y)

        y_pred = model.predict(X)

        print(le.inverse_transform(y_pred))  

        print("Predicting used {}".format(name))

        print(accuracy_score(y,y_pred))

        print(confusion_matrix(y, y_pred))
