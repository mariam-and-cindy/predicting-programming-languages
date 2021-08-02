import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from io import StringIO
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from IPython.display import display, display_html 

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE


# imports for modeling
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier





def nlp_xy_split(X_data, y_data):
    '''
    This function  split data during  NLP.
    X_data : cv = CountVectorizer()
            X = cv.fit_transform(df.lemmatized)
    y_data : target (df.target)
    
    Returns : X_train, y_train, X_validate, y_validate, X_test, y_test
    
    Example :
    X_train, y_train, X_validate, y_validate, X_test, y_test = nlp_xy_split (X,y)
    '''
    from sklearn.model_selection import train_test_split
    X_train_validate, X_test, y_train_validate, y_test = train_test_split(X_data, y_data, 
                                                                          stratify = y_data, 
                                                                          test_size=.2, random_state=123)
    
    X_train, X_validate, y_train, y_validate = train_test_split(X_train_validate, y_train_validate, 
                                                                stratify = y_train_validate, 
                                                                test_size=.3, 
                                                                random_state=123)
    
    print(f'X_train -> {X_train.shape}               y_train->{y_train.shape}')
    print(f'X_validate -> {X_validate.shape}         y_validate->{y_validate.shape} ')        
    print(f'X_test -> {X_test.shape}                  y_test>{y_test.shape}') 


    return X_train, y_train, X_validate, y_validate, X_test, y_test



def model_performs (X_df, y_df, model):
    '''
    Take in a X_df, y_df and model  and fit the model , make a prediction, calculate score (accuracy), 
    confusion matrix, rates, clasification report.
    X_df: train, validate or  test. Select one
    y_df: it has to be the same as X_df.
    model: name of your model that you prevously created 
    
    Example:
    mmodel_performs (X_train, y_train, model1)
    '''

    #prediction
    pred = model.predict(X_df)

    #score = accuracy
    acc = model.score(X_df, y_df)

    #conf Matrix
    conf = confusion_matrix(y_df, pred)
    mat =  pd.DataFrame ((confusion_matrix(y_df, pred )),index = ['a_JavaScript','a_Python', 'a_Java', 'a_C++', 'a_TypeScript' ],\
     columns =['p_JavaScript','p_Python', 'p_Java', 'p_C++', 'p_TypeScript' ])
    #rubric_df = pd.DataFrame([['True Negative', 'False positive'], ['False Negative', 'True Positive']], columns=mat.columns, index=mat.index)
    #cf = rubric_df + ': ' + mat.values.astype(str)

    #assign the values
    tp = conf[1,1]
    fp =conf[0,1] 
    fn= conf[1,0]
    tn =conf[0,0]

    #calculate the rate
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    tnr = tn/(tn+fp)
    fnr = fn/(fn+tp)

    #classification report
    clas_rep =pd.DataFrame(classification_report(y_df, pred, output_dict=True)).T
    clas_rep.rename(index={'0': "No Aproved", '1': "Approved"}, inplace = True)
    print(f'''
    The accuracy for our model is {acc:.4%}

    ________________________________________________________________________________
    
    Classification Report:
    ''')
    display(clas_rep)

    return round(acc, 3)


def compare_train_val (model, name_dataset1, X, y, name_dataset2,  X2, y2 , name):
    '''
    Take in a  model and compare the  performance metrics of  Train, Evaluate  (only 2).
    model: the model that you want to compare
    name_dataset1 : type :train, validate or  test. Select one, STRING
    X: df test, validate or test
    y: df test, validate or test
    name_dataset2: type :train, validate or  test. Select one, STRING
    X2: df2 test, validate or test
    y2: df2 test, validate or test
    name:model name
    
    Example:
    compare_metrics(logit2,'Train',X_train, y_train,'Test', X_test, y_test, 'dec_tree')
    '''
    from IPython.display import display, display_html
    if name_dataset1.lower() != "train" and name_dataset1.lower() != "validate" and name_dataset1.lower() != "test" :
        return print("incorrect name")
    if name_dataset2.lower() != "train" and name_dataset2.lower() != "validate" and name_dataset2.lower() != "test" :
        return print("incorrect name")
    #prediction
    pred_1 = model.predict(X)
    pred_2 = model.predict(X2)

    #score = accuracy
    acc_1 = round(model.score(X, y),3)
    acc_2 = round(model.score(X2, y2),3)


    #conf Matrix
    #model 1

    conf_1 = confusion_matrix(y, pred_1)
    cf_1 =  pd.DataFrame ((confusion_matrix(y, pred_1 )),index = ['a_JavaScript','a_Python', 'a_Java', 'a_C++', 'a_TypeScript' ],\
    columns =['p_JavaScript','p_Python', 'p_Java', 'p_C++', 'p_TypeScript' ])
    
    #model2
    conf_2 = confusion_matrix(y2, pred_2)
    cf_2 =  pd.DataFrame ((confusion_matrix(y2, pred_2 )),index = ['a_JavaScript','a_Python', 'a_Java', 'a_C++', 'a_TypeScript' ],\
    columns =['p_JavaScript','p_Python', 'p_Java', 'p_C++', 'p_TypeScript' ])
    #

    #classification report
    #model1
    clas_rep_1 =pd.DataFrame(classification_report(y, pred_1, output_dict=True)).T
    clas_rep_1.rename(index={'0': "dead", '1': "survived"}, inplace = True)

    #model2
    clas_rep_2 =pd.DataFrame(classification_report(y2, pred_2, output_dict=True)).T
    clas_rep_2.rename(index={'0': "dead", '1': "survived"}, inplace = True)
    print(f'''
    ******    {name_dataset1}       ******                              ******     {name_dataset2}    ****** 
       Overall Accuracy:  {acc_1:.2%}              |                Overall Accuracy:  {acc_2:.2%}  
                                                

    _________________________________________________________________________________
    ''')
  
    cf_1_styler = cf_1.style.set_table_attributes("style='display:inline'").set_caption(f'{name_dataset1} Confusion Matrix')
    cf_2_styler = cf_2.style.set_table_attributes("style='display:inline'").set_caption(f'{name_dataset2} Confusion Matrix')
    space = "\xa0" * 25
    display_html(cf_1_styler._repr_html_()+ space  + cf_2_styler._repr_html_(), raw=True)
    print('''
    ________________________________________________________________________________
    
    Classification Report:
    ''')
    clas_rep_1_styler = clas_rep_1.style.set_table_attributes("style='display:inline'").set_caption(f'{name_dataset1} Classification Report')
    clas_rep_2_styler = clas_rep_2.style.set_table_attributes("style='display:inline'").set_caption(f'{name_dataset2} Classification Report')
    space = "\xa0" * 45
    display_html(clas_rep_1_styler._repr_html_()+ space  + clas_rep_2_styler._repr_html_(), raw=True)
    
    
    metric_dic = {'model_name': name, 
          (name_dataset1 +'_score'): acc_1,
          (name_dataset2 +'_score'): acc_2}
    return metric_dic


def whole_model (mod, X_train, y_train, X_validate, y_validate, model_name, metric_df):
    '''
    this function fit model , calculate metrics for train, validate  and return a df with score of the model
    Example:
    metric_df = whole_model ((LogisticRegression(C=1.0 , random_state=123)), X_train, y_train, X_validate, y_validate, 'log_reg_ngrams', metric_df)
    '''
    #fit
    modelr =mod.fit(X_train, y_train)
    #calculate metrics
    res = compare_train_val(modelr,'train',X_train, y_train,'validate', X_validate, y_validate, model_name)
    metric_df = metric_df.append(res, ignore_index = True)
    return metric_df
