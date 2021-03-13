import sys
import os
import numpy as np
import numpy.random as npr
import pandas as pd
import random

#imports for part 1
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import string

#imports for part 3
from sklearn.decomposition import TruncatedSVD
import io

#imports for bonus b
from sklearn.svm import LinearSVC


# Code for Part 1
def preprocess(inputfile):
    pre_processed = []
    for line in inputfile.readlines():
        old_word = WhitespaceTokenizer().tokenize(line)[2] 
        new_word = WordNetLemmatizer().lemmatize(old_word).lower() #lemmatize and lowercase
        if new_word not in stopwords.words() and new_word not in string.punctuation: #remove stopwords and puncts
            pre_processed.append(line.replace(old_word, new_word))
    return pre_processed

# Code for part 2
class Instance:
    def __init__(self, neclass, features):
        self.neclass = neclass
        self.features = features

    def __str__(self):
        return "Class: {} Features: {}".format(self.neclass, self.features)

    def __repr__(self):
        return str(self)

def create_instances(data):
    instances = []
    for i, line in enumerate(data):
        entity = line.split('\t')[4]
        if "B" in entity: # the beginning of an NE
            end_i = i
            while  data[i].split('\t')[4].split('-')[1] in data[end_i].split('\t')[4]:
                end_i += 1
            neclass = entity.split('-')[1] #get the NE class
            features = []
            #get the context 
            entity_sent_nr = data[i].split('\t')[1]
            count = 1
            for j in range(i-5,i): #5 words before the beginning of the NE
                if j < 0 or "B-" in data[j].split('\t')[4] or "I-" in data[j].split('\t')[4]:
                    pass  #skipping NEs and index out of range 
                else:
                    sent_nr = data[j].split('\t')[1]
                    if sent_nr == entity_sent_nr:
                        word = data[j].split('\t')[2]
                        features.append(word)
                    else:
                        s_tok = '<S'+ str(count) +'>'
                        features.append(s_tok)
                        count += 1 
            count = 1
            for k in range(end_i,end_i+6): #5 words after the end of the NE
                if k >= len(data) or "B-" in data[k].split('\t')[4] or "I-" in data[k].split('\t')[4]:
                    pass
                else:
                    sent_nr = data[k].split('\t')[1]
                    if sent_nr == entity_sent_nr:
                        word = data[k].split('\t')[2]
                        features.append(word)
                    else:
                        e_tok = '<E'+ str(count) +'>'
                        features.append(e_tok)
                        count += 1 
#             ne = data[i].split('\t')[2]
#             print(ne," - ",entity_sent_nr," - ", features)
            instances.append(Instance(neclass, features))

    return instances

# Code for part 3
def create_table(instances):
    #get all words in the document
    all_feats = []
    for instance in instances:
        all_feats = all_feats + instance.features
    header = ['Class'] + all_feats
    df = pd.DataFrame(columns=header)

    #get the unique words with counts and create word count matrix
    for i, instance in enumerate(instances):
        row = [instance.features.count(feat) for feat in all_feats]
        df.loc[i] = [instance.neclass.replace('\n','')] + row
        
    #reduce it.
    ne_classes = df['Class']
    temp_df = df.drop('Class', 1) #drop the class column to not include it in the dim reduction
    red_df = reduce(temp_df)
    red_df.insert(loc = 0, column = 'Class', value = ne_classes) #insert the class column back

    return red_df


def reduce(matrix, dims=1000):
    svd = TruncatedSVD(dims)
    transformed = svd.fit_transform(matrix)
    reduced = pd.DataFrame(data=transformed)
    return reduced

def ttsplit(bigdf):
    threshold = int( len(bigdf.index) * 80 / 100 )
    random_bigdf = bigdf.sample(frac=1).reset_index(drop=True) # shuffle the rows
    
    df_train = random_bigdf.iloc[:threshold, :] # split on threshold
    df_train_classes = df_train['Class']
    df_train = df_train.drop('Class', 1)
    
    df_test = random_bigdf.iloc[threshold:, :]
    df_test_classes = df_test['Class']
    df_test = df_test.drop('Class', 1)

    return df_train.to_numpy(), df_train_classes.to_numpy(), df_test.to_numpy(), df_test_classes.to_numpy()

# Code for part 5
def confusion_matrix(truths, predictions):
    
    unique_classes = np.unique(np.concatenate((truths, predictions), axis=0)) #get all the possible classes
    no_classes = len(unique_classes)
    matrix = pd.DataFrame(data =np.zeros((no_classes,no_classes)),index= unique_classes, columns = unique_classes) #initialize the df
        
    for i in range(len(truths)):
        matrix.at[truths[i], predictions[i]] += 1
        
    return matrix



# Code for bonus part B
def create_instances_with_pos(data):
    instances = []
    for i, line in enumerate(data):
        entity = line.split('\t')[4]      
        if "B" in entity:
            end_i = i
            while  data[i].split('\t')[4].split('-')[1] in data[end_i].split('\t')[4]:
                end_i += 1
            neclass = entity.split('-')[1] #get the NE class
            features = []
            #get the context 
            entity_sent_nr = data[i].split('\t')[1]
            count = 1
            for j in range(i-5,i): #5 words before
                if j < 0 or "B-" in data[j].split('\t')[4] or "I-" in data[j].split('\t')[4]:
                    pass  #skipping NEs and index out of range 
                else:
                    sent_nr = data[j].split('\t')[1]
                    if sent_nr == entity_sent_nr:
                        word = data[j].split('\t')[2]
                        pos = data[j].split('\t')[3]
                        word_and_pos = word +"-"+ pos
                        features.append(word_and_pos)
                    else:
                        s_tok = '<S'+ str(count) +'>'
                        features.append(s_tok)
                        count += 1 
            count = 1
            for k in range(end_i,end_i+6): #5 words after
                if k >= len(data) or "B-" in data[k].split('\t')[4] or "I-" in data[k].split('\t')[4]:
                    pass
                else:
                    sent_nr = data[k].split('\t')[1]
                    if sent_nr == entity_sent_nr:
                        word = data[k].split('\t')[2]
                        pos = data[j].split('\t')[3]
                        word_and_pos = word +"-"+ pos
                        features.append(word_and_pos)
                    else:
                        e_tok = '<E'+ str(count) +'>'
                        features.append(e_tok)
                        count += 1 
#             ne = data[i].split('\t')[2]
#             print(ne," - ",entity_sent_nr," - ", features)
            instances.append(Instance(neclass, features))

    return instances



def bonusb(filename):
    #process raw data
    gmbfile = open(filename, "r")
    inputdata = preprocess(gmbfile)
    gmbfile.close()
    
    #create instances
    instances = create_instances_with_pos(inputdata)
    
    #reduce
    bigdf = create_table(instances)
    
    #train
    train_X, train_y, test_X, test_y = ttsplit(bigdf)
    model = LinearSVC()
    model.fit(train_X, train_y)
    train_predictions = model.predict(train_X)
    test_predictions = model.predict(test_X)
    
    #evaluate
    train_matrix = confusion_matrix(train_y, train_predictions)
    test_matrix = confusion_matrix(test_y, test_predictions)
    
    return train_matrix, test_matrix
    
