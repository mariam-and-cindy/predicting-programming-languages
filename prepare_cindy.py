import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd
import acquire as a
from textblob import TextBlob


#import
import warnings
warnings.filterwarnings("ignore")





def clean_df(df):
    '''
    takes in a df and lowercase everything, normalize unicode characters, replace anything that is not a letter,
    number, whitespace or a single quote.
    retunr a clean df
    '''
    for column in df.columns:
        #get lowercase
        df[column]= df[column].str.lower()
        #normalize
        df[column]= [unicodedata.normalize('NFKC',df[column][n])\
                     .encode('ascii', 'ignore')\
                    .decode('utf-8') for n in range (0, len(df))]
        #remove special characters
        df[column]= [re.sub(r"[^a-z0-9\s]", '', df[column][n] ) for n in range (0, len(df))]
    return df




def basic_clean (string):
    '''
    takes in a string and lowercase everything, normalize unicode characters, replace anything that is not a letter,
    number, whitespace or a single quote.
    retunr a clean string
    '''
    
    string = string.lower()
    string = unicodedata.normalize('NFKC',string)\
    .encode('ascii', 'ignore')\
    .decode('utf-8')
    string = re.sub(r"[^a-z0-9\s]", '', string)
    return string


def tokenize (string):
    '''
    take in a string and tokenize all the words in the string
    '''
    
    # Create the tokenizer
    tokenizer = nltk.tokenize.ToktokTokenizer()
    # Use the tokenizer
    string = tokenizer.tokenize(string, return_str = True)
    return string


def stem (string):
    '''
     takes in a text and return the text after applying stemming to all the words.
    '''
    # Create porter stemmer.
    ps = nltk.porter.PorterStemmer()
    # Apply the stemmer to each word in our string.
    stems = [ps.stem(word) for word in string.split()]
    text_stemmed = ' '.join(stems)
    return text_stemmed 



def lemmatize (text):
    '''
     Takes in some text and return the text after applying lemmatization to each word.
    '''
    wnl = nltk.stem.WordNetLemmatizer()
    
    # Use the lemmatizer on each word in the list of words we created by using split.
    lemmas = [wnl.lemmatize(word) for word in text.split()]
    
    # Join our list of words into a string again; assign to a variable to save changes.
    text_lemmatized = ' '.join(lemmas)
    
    return text_lemmatized
    


def remove_stopwords (string, extra_words= [], exclude_words=[]):
    '''
    Takes in a strand return the text after removing all the stopwords.
    Parameters:
    string : text in  string type
    extra_words : list of additional stop words to include,
    exclude_words : list of any words that we don't want to remove.
    '''
    # standard English language stopwords list from nltk
    stopword_list = stopwords.words('english')
    
    #add extra_words
    stopword_list =stopword_list + extra_words
    
    #exclude words
    for element in exclude_words:
        if element in stopword_list:
            stopword_list.remove(element)
    # Split words.
    words = string.split()
    
    # Create a list of words from my string with stopwords removed and assign to variable.
    filtered_words = [word for word in words if word not in stopword_list]
    
    # Join words in the list back into strings; assign to a variable to keep changes.
    string_without_stopwords = ' '.join(filtered_words)
    
    return string_without_stopwords



def prepare_data(df, column, extra_words=[], exclude_words=[]):
    '''
    This function take in a df and the string name for a text column with 
    option to pass lists for extra_words and exclude_words and
    returns a df with the  original text, cleaned (tokenized and stopwords removed) , stemmed text,
    lemmatized text.
    '''
    df['clean'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    
    df['stemmed'] = df['clean'].apply(stem)
    
    df['lemmatized'] = df['clean'].apply(lemmatize)
    
    return df




def miss_dup_values(df):
    '''
    takes in a dataframe of observations and attributes and returns a dataframe where each row is an atttribute name, 
    the first column is the number of rows with missing values for that attribute, 
    and the second column is percent of total rows that have missing values for that attribute and 
    duplicated rows.
    '''
        # Total missing values
    mis_val = df.isnull().sum()
        # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
        #total of duplicated
    dup = df.duplicated().sum()  
        # Percentage of missing values
    dup_percent = 100 * dup / len(df)
        # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
    mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
           "There are " + str(mis_val_table_ren_columns.shape[0]) +
           " columns that have missing values.")
    print( "  ")
    print (f"** There are {dup} duplicate rows that represents {round(dup_percent, 2)}% of total Values**")
        # Return the dataframe with missing information
    return mis_val_table_ren_columns


def remove_nonenglish (df):
    '''
    takes in df and 1 column to check if the text is in englis if not that row is going to be remove
    '''
    for n in range (0, len(df)):
        text = df.readme_contents[n]
        lang = TextBlob(text)
        if lang.detect_language() != 'en':
            df =df.drop([n])
    return df



def prepare_mf (df):
    '''
    takes in a df and all the rows with missing information, non English text,
    and then clean, tokenize, stemming, lemmatize
    '''
        #removing missing values
    df = df.dropna(axis=0).reset_index(drop=True)
    #removing texts that are not English
    df = remove_nonenglish(df)


    #use my prepare function to  clean, tokenized, stemming, lemmatize
    df =prepare_data(df, 'readme_contents')
    return df
    