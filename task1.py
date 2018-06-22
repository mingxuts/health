'''
Created on Jun 21, 2018

@author: longgu
'''

import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
from sklearn import linear_model
import nltk

MAX_NUM_WORDS = 10000

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
    data_df = data_df[data_df.Investigated.isin(['No',"Yes"])]
    
    #remove data whose description is -
    data_df = data_df[data_df['Published Event Description (DAEN)']!='-']
    
    #Transform Yes and No to 1 and 0
    data_df.loc[data_df.Investigated =='No','Investigated'] = 0
    data_df.loc[data_df.Investigated =='Yes','Investigated'] = 1
    
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

if __name__ == "__main__":
    
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = nltk.corpus.stopwords.words('english')
    
    print('downloading nltk')
    
    data_df = data_processing()
    
    data_df_test = data_df
     
    words, words_vecmap = read_glove_vecs('../data/glove.6B.50d.txt')
    dictionary = list(words_vecmap.keys())
    
    print("loading word vec")
    
    X = model_embedding(data_df_test,words_vecmap,dictionary,stop_words)
    print( X )
    print(X.shape)
    y = data_df_test.Investigated.astype(int)
    print(y)
     
    C = 1.0
 
    classifier = linear_model.LogisticRegression(C=C)
 
    classifier.fit(X, y)
 
    y_pred = classifier.predict(X)    
     
    print(accuracy_score(y,y_pred))
    print(confusion_matrix(y, y_pred))
    
#     print(words_vecmap['epistaxis'])
    
    print(data_df.shape)
