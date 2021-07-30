# Predicting Programming Languages from Github Repositories  - NLP Project 

 

## Project Description

-------------------

Perform statistical analysis on data collected via web-scraping of the GitHub site. After collecting a minimal amount of data from GitHub, including the project's Readme file, attmept to predict the language used in each project.

 

This involves data cleaning, wrangling and exploration, as well as modeling and validation/verification of model result.

 

## Project Goals

-------------

1. Create scripts to perform the following:

                a. acquisition of data from GitHub's website

                b. preparation of data

                c. exploration of data

2. Perform statistical analysis to test hypotheses

3. Build and evaluate Classification models to predict the Programming Language used in a given project

 

## Business Goals

--------------

- Make use of NLP to predict programming language based on Readme documents

- Perform a number of parsing operations to isolate and process key text features - including lemmatization, stemming and removal of stopwords

 

Initial Hypotheses

------------------

Hypotheses 1:

 

Confidence level =

Alpha = 1 - Confidence level =

H0:

H1:

Hypotheses 2:

 

## Data Dictionary

---------------

Name    Datatype             Definition            Possible Values

repo name          non-null string   Unique name for the repo            slash delimited string

repo language   non-null string   The programming language used in this project  string (eg python/javascript/etc)

readme contents             non-null string   The entirety of the project's readme file plaintext string

 

Additionally, a set of features were added to the data set:

 

Name    Datatype             Definition            Possible Values

stemmed           

lemmatized

etc

 

## Project Planning

----------------

The overall process followed in this project, is as follows:

 

Plan

Acquire

Prepare

Explore

Model

Deliver

 

### 1. Plan

Create a list of tasks to complete in the Trello Board

Perform preliminary examination of a number of GitHub projects

Acquire tokens and permissions to scrape data from the GitHub website

 

### 2. Acquire

This is accomplished via the python script named “acquire.py”. The script will use credentials (stored in env.py) to collects data using from GitHub.com in various ways

- first, collect a number of "URLs", or Repository names, so that the subsequent acquision function will be able to seek out those repositories

- use BeautifulSoup to parse multiple pages (approximately 20) of the top-forked repositories, captured here: https://github.com/search?o=desc&p=<pageNumber>&q=stars%3A%3E1&s=forks&type=Repositories

- apply random sleep durations to web calls to avoid rate limiting - GitHub prevents certain high-frequency requests

- store these in git_urls.csv - that way, we would not hit GituHub's page and scrape the same data repeatedly. Moreover, this ensures that subsequent processing executions will consistently use the same repo list, leading to a more reliable and consistent result

- capture these repository names as a strings in  a python list object

- Once the list of repositories is collected, use a second script to collect the following information from those repositories, including:

                - repository name

                - actual language of the project

                - contents of the readme for that repository

- These values are dumped, as json data, into the data2.json file - this once again ensures that we are not using cached data and avoids unnecessary web calls to the GitHub API, while maintaining consistency in results

 

### 3. Prepare

This functionality is stored in the python script "prepare.py". It will perform the following actions:

- lowercase the readme contents to avoid case sensitivity

- remove non-standard (non ascii) characters

- tokenize the data

- applying stemming

- apply lemmatization

- remove unnecessary stopwords

- remove non-english records (some repositories are written in other languages altogether)

- remove any records where the readme contents was null or empty

- Split the data into 3 datasets - train/test/validate - used in modeling

Train: 56% of the data

Validate: 24% of the data

Test: 20% of the data

 

### 4. Explore

This functionality resides in the "explore.py" file, which provides the following functionality:

 

 

### 5. Model

Generate a baseline, against which all models will be evaluated

Compare the models against the baseline and deduce which has the lowest RMSE and highest R-squared value

Fit the best performing model on test data

Create visualizations of the residuals and the actual vs predicted distributions

 

### 6. Deliver

Present findings via PowerPoint slides


### To recreate

Simply clone the project locally and create an env.py file in the same folder as the cloned code. The env.py file's format should be as follows:

 
github_token = 'GITHUB-TOKEN'

github_username = 'GITHUB-USERNAME'


Next, run the acquire script in your command line, using the following command:

python acquire.py

 

Finally, open the Jupyter notebook titled “” and execute the code within.

 


### Takeaways

---------

During the analysis process, we made use of the following classification  models:

 

 

### Next Steps

----------

If we had more time, we would:
