import sys
import re

import numpy as np
import pickle
import pandas as pd
import sqlalchemy
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def load_data(database_filepath:str):
    '''Loads a sqlite database file into a pandas DataFrame and splits into predictor and target variables

    Args:
        database_filepath (str): sqlite file of the database

    Returns:
        X (pd.DataFrame): predictor variables
        y (pd.DataFrames): target variables
        categories (list): list of prediction classes 
    '''
    engine = sqlalchemy.create_engine("sqlite:///%s" % database_filepath)
    df = pd.read_sql_table('disaster_messages', engine)
    X = df.message.copy()
    y = df[df.columns[4:]].copy()
    y = y.apply(pd.to_numeric)
    categories = df.columns[4:].copy()
    return X, y, categories


def tokenize(text:str):
    '''Transforms text into list of tokens.
    Args:
        text (str): text to be processed
    
    Returns:
        clean_tokens (list): list of normalized tokens
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(verbose:bool=False):
    ''' Builds ML pipeline for cross validation multioutput process

    Args:
        verbose (boolean): if true prints the pipeline params keys.

    Returns:
        (sklearn.Predictor): model ready to train.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(MLPClassifier()))
    ])
    if verbose:
        for p in pipeline.get_params().keys():
           print(p)

    parameters = {
        'tfidf__norm': ['l2', 'l1'],
        'clf__estimator__hidden_layer_sizes': [
            (50,),
            (50, 25),
        ],
        'clf__estimator__learning_rate_init': [
            0.001,
            0.01
        ]
    }

    cv = GridSearchCV(pipeline, parameters, verbose=1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluates trained model per each category and prints results in standard output.

    Args:
        model (sklearn.Predictor): trained sklearn model
        X_test (pd.DataFrame): testing predictor vraiables
        Y_test (pd.DataFrame): testing target variables
        category_names (list): list of classification categories
    
    Returns:
        None
    '''
    print("\nBest Parameters:", model.best_params_)
    y_pred = model.predict(X_test)
    labels = np.unique(category_names)
    accuracy = (y_pred == Y_test).mean()

    print(accuracy)

    for i, category in enumerate(category_names):
        print(category, ':')
        print(classification_report(Y_test[category], y_pred[:, i]))
        print()


def save_model(model, model_filepath):
    '''Saves the model into pickle format.
    
    Args:
        model (sklearn.Predictor): sklearn model
        model_filepath (str): filepath to store the model

    Returns:
        None 
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''Runs the whole ML pipeline as follows: loads data, builds, trains, evaluate and saves model.
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
