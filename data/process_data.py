import sys

import pandas as pd
import sqlalchemy


def load_data(messages_filepath:str, categories_filepath:str) -> pd.DataFrame:
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


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
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
    # drop observations in related column so it is left with 1's and 0's
    df.drop(df[df['related'] == 2].index, inplace = True)
    
    return df


def save_data(df:pd.DataFrame, database_filename:str) -> None:
    """ saves a dataframe in a sqlite database
    
    Args:
        df (pd.DataFrame): data to be saved
        database_filename (str): filename to save the database
    Returns:
        None
    """
    engine = sqlalchemy.create_engine("sqlite:///%s" % database_filename)
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')


def main() -> None:
    """ runs the whole ETL pipeline where text data from tweet messages and categories
    assigned to each one are loaded, processed and saved in a sqlite database
    """
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
