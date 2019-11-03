# Import necessary Python modules/packages

import sys
import nltk
import re
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import numpy as np
import datetime
from pytz import timezone


import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline


def load_data(database_filepath):
    
    '''
    Load data from database as dataframe
    Input: database_filepath: File path of sql database

    Output:

        X: Message data (predictor variables/features)
        Y: Categories (target/response  variables)
        category_names: Labels for 36 categories

    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM disaster_response", engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    
    return X, Y, category_names


def tokenize(text):
    
    '''
    Tokenize and clean text
    Input: original message text

    Output:clean_tokens:

    '''
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())

    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove Stopwords
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
  
    '''
    Build a machine learning  Pipeline Model using 3 Estimators(Vectorizer, TFid, Multioutput Classifiere(Random Forest) and  improve
    it using gridsearch
    Input: None
    Output: Improved Model

    '''
    # Create Pipeline for 3 estimators(Vectorizer, TFid, Multioutput Classifier
    pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    # Set  Hyperestimator values for GridSearchCV
    
    parameters = {'clf__estimator__n_estimators': [50],
                  'clf__estimator__min_samples_split': [2, 4, 6],
                  'clf__estimator__max_depth': [2,5,10],
                  'clf__estimator__min_samples_leaf': [3,4,5]
                  
                 }

    pipeline_grid = GridSearchCV(pipeline, param_grid= parameters,cv= 3, n_jobs=4)
    
    #print(pipeline_grid.get_params())
    
    return pipeline_grid

       
    
def evaluate_model(model, X_test, Y_test, category_names):
   
    '''
    Evaluate model performance using Test data

    Input: 
        model: Model to be evaluated
        X_test: Test data (predictor variables)
        Y_test: Actual lables/response variables for Test data
        category_names: Labels for 36 categories

    Output:
        Classfication report elements(Precision, Recall, F1 score)  and accuracy score for each category

    '''

    # predict on test data

    y_pred = model.predict(X_test)
    print(type(y_pred))

    # display Accuracy Report

    for i in range(len(category_names)):

        print("Message Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy of  Category: %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, y_pred[:,i])))

        

def save_model(model, model_filepath):
    
    '''
    Save model as a pickle file 
    Input: 
        model: Model to be saved
        model_filepath: path of the output pick file

    Output:
        A pickle file of saved model

    '''
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    
    #To run ML pipeline that trains classifier and saves
    #    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    
    tz = timezone('EST')
    start_time  = datetime.datetime.now(tz)
    print('Model Execution start time...', start_time) 

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print("X Shape", X_train.shape)
        print("Y Shape", Y_train.shape)
        
        
        print('Building model...')
        model = build_model()
        
       
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        
        end_time = datetime.datetime.now(tz)
        print('Model Execution End Time',end_time) 
        print('Model Execution Duration(in Secs) ',(end_time-start_time).seconds) 

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()