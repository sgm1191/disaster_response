import sys

import pandas as pd

def load_data(messages_filepath:str, categories_filepath:str):
    """ loads disaster data from csv files

    Args:
        messages_filepath (str): path of disaster messages csv file
        categories_filepath (str): path of disaster categories csv file

    Returns:
        pd.DataFrame: a dataframe with merged data of messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories  = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id")
    return df


def clean_data(df):
    """ processes categories column and splits it into one column for each category
    in a one-hot like way, also removes duplicates.

    Args:
        df (pd.DataFrame): dataframe of disaster messages and categories

    Returns:
        pd.DataFrame: processed data frame
    """
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].astype(str).str.split('-').str.get(1)
    categories[column] = pd.to_numeric(categories[column])

    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    with create_engine('sqlite:///tweets.db') as engine:
        df.to_sql('disaster_messages', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()