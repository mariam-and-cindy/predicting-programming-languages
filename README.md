# Predicting Programming Languages from Github Repositories  - NLP Project 

 

## Project Description

-------------------

Perform statistical analysis on data collected via web-scraping of the GitHub site. After collecting over 300 repositories from GitHub we predicted the primary programming language used in each project.

 

This project involves data cleaning, wrangling and exploration, as well as modeling and validation/verification of modeling results.

 

## Project Goals

-------------

1. Create scripts to perform the following:

                a. acquisition of data from GitHub's website

                b. preparation of data

                c. exploration of data

                d. modeling

2. Build and evaluate Classification models to predict the Programming Language used in a given Readme.

 

## Business Goals

--------------

- Make use of NLP to predict programming language based on Readme content.

- Perform a number of parsing operations to isolate and process key text features - including lemmatization, stemming and removal of stopwords.

 


 

## Data Dictionary

---------------

| Name |   Datatype   |      Definition    |    Possible Values  |
| ----- | ----- | ----- |----- |
| repo  |     object  | Unique name for the repo |  slash delimited string|
| language |  object | The programming language used in this project | string (eg python/javascript/etc) |
| readme contents     |   object |  The entirety of the project's readme file | plain text string |

 

Additionally, a set of features were added to the data set:

 

| Name  |  Datatype    |         Definition     |       Possible Values |
| ----- | ----- | ----- |----- |
| clean |  object|  Parsing Text of the readme_content column  :| plain text string  |
| stemmed |  object |  Stemmed text of the clean column | plain text string  |
| lemmatized | object| Lemmatized text of clean  column | plain text string  |
| readme_length |  int64  | lenght of the README content |  numeric |
| word_count | int64  | total of words of the README  | numeric |


 

## Project Planning

----------------

The overall process followed in this project, is as follows:

 

###  Plan  -->  Acquire   --> Prepare  --> Explore  --> Model  --> Deliver


--------------

### 1. Plan

Create a list of tasks to complete in [this Trello Board.](https://trello.com/b/PddXdOTJ/nlp-project)

Perform preliminary examination of a number of GitHub projects.

Acquire tokens and permissions to scrape data from the GitHub website.

 

### 2. Acquire

This is accomplished via the python script named “acquire.py”. The script will use credentials (stored in env.py) to collect data from GitHub.com in various ways

- first, collect a number of "URLs", or Repository names, so that the subsequent acquision function will be able to seek out those repositories.

- use BeautifulSoup to parse multiple pages (approximately 20) of the top-forked repositories, captured [here.](https://github.com/search?o=desc&p=<pageNumber>&q=stars%3A%3E1&s=forks&type=Repositories)

- apply random sleep durations to web calls to avoid rate limiting - GitHub prevents certain high-frequency requests.

- store these in git_urls.csv - that way, we would not hit GituHub's page and scrape the same data repeatedly. Moreover, this ensures that subsequent processing executions will consistently use the same repo list, leading to a more reliable and consistent result.

- capture these repository names as a strings in a python list object.

- Once the list of repositories is collected, use functions from the acquire script to collect the following information from those repositories, including:

                - repository name

                - actual language of the project

                - contents of the readme for that repository

- These values are dumped, as json data, into the data2.json file - this once again ensures that we are not using cached data and avoids unnecessary web calls to the GitHub API, while maintaining consistency in results.

- After removing non English repositories from the json file, we cache it as a csv file and read that into a pandas dataframe in this notebook.

 

### 3. Prepare

This functionality is stored in the python script "prepare.py". It will perform the following actions:

- lowercase the readme contents to avoid case sensitivity

- remove non-standard (non ascii) characters, any accented characters

- tokenize the data

- applying stemming

- apply lemmatization

- remove unnecessary stopwords

- remove any records where the readme contents were null or empty

- Split the data into 3 datasets - train/test/validate - used in modeling

  - Train: 56% of the data

  - Validate: 24% of the data

  - Test: 20% of the data

 

### 4. Explore

This functionality resides in the "explore.py" file, which provides the following functionality:

- What are the most common words in READMEs?
 
- What does the distribution of IDFs look like for the most common words?
 
- Does the length of the README vary by programming language?
 
- Do different programming languages use a different number of unique words?
 



### 5. Model

Generate a baseline, against which all models will be evaluated.

Compare the models against the baseline and deduce which has the highest overall accuracy scores.

Fit the best performing model on test data.



### 6. Deliver

Present findings via PowerPoint slides.


### To recreate

Simply clone the project locally and create an env.py file in the same folder as the cloned code. The env.py file's format should be as follows:

 
github_token = 'GITHUB-TOKEN'

github_username = 'GITHUB-USERNAME'


Next, run the acquire script in your command line, using the following command:

python acquire.py

A file named 'data2.json' will be stored locally in this repository. You can either convert this file to a csv and read that into a dataframe or download the git_data.csv file in this repository and read that into a dataframe.
 
 
Finally, open the Jupyter notebook titled “final_report_nlp” and execute the code within.

 


### Takeaways

---------

- During the analysis process, we made use of the following classification  models:
  - Decsion Tree
  - KNN
  - Logistic Regression
  - Random Forest
  - Naive Bayes
- We evaluated model perfomance based on the overall accuracy scores.
- All models beat the baseline accuracy of 34%.
- OUr best performing model was the Decision Tree Model which had an overall accuracy of 55% on test data.
 

 

### Next Steps

----------

If we had more time, we would:

- Use idf to create models instead of bag of words
- Use fewer languages in models

