import sys
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
  
    '''

    Input:

        messages_filepath: File path of messages data(e.g.'data/disaster_messages.csv')
        categories_filepath: File path of categories data(e.g, 'data/disaster_categories.csv')

    Output:

        df: Merged dataset from messages and categories

    '''
       
    # read message data
    messages = pd.read_csv(messages_filepath)
    messages.head()

    # read categories data
    categories = pd.read_csv(categories_filepath)
    categories.head()
    
    # merge datasets (Messages and Categories)
    df = messages.merge(categories, how='outer',on=['id'])
    df.head()
    
    return df


def clean_data(df):

    '''
    Input:
        Merged Data frame from messagaes and categories files
    Output:

        df: Cleaned Dataset

    '''
    
    # create a dataframe of the 36 individual category columns

    categories = df['categories'].str.split(';',expand=True)
    categories.head()

    # select the first row of the categories dataframe
    row = categories.iloc[0]
          
    # create new column names using 1st row of categories 
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of categories
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
    
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-').str[1]
       # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    
    # drop the original categories column from `df`
    df.drop(['categories'],inplace = True, axis = 1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # Drop Duplicates
    df.drop_duplicates(subset='id', inplace=True)
    
    # Clean-up Data Frame- Shape
    print( "Rows =", df.shape[0], "Columns =", df.shape[1])
    
    return df
    
    
def save_data(df, database_filename):
    
    '''
    Save df into sqlite db

    Input:

        df: cleaned dataset
        database_filename: database name

    Output: 

        A SQLite database

    '''
    
    # save the clean data set into Sqlite Database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_response', engine, if_exists='replace',index=False)
       
    # Read from Sqlit database
    print(' Load sample data from Sqlite Database...')
    df_read =pd.read_sql("SELECT * FROM disaster_response", engine)
    print(df_read.head(10))

def main():
 
    #To run ETL pipeline that cleans data and stores in database
    #    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    
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