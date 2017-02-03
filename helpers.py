import math
import pandas as pd
import re
import numpy as np
import nltk

def get_score(ar):
  nltk.data.path.append('./nltk_data/')
  score = predict_score(ar)
  return score

def make_array(x):
    dat = []

    dat.append(contains_currency(x))
    dat.append(contains_symbol_separators(x))
    dat.append(contains_question(x))
    dat.append(contains_quotes(x))
    dat.append(contains_weird_symbols(x))
    dat.append(find_num_words(x))
    dat.append(find_longest_word_length(x))
    dat.append(find_charlength(x))
    dat.append(contains_dotcom(x))
    dat.append(contains_percentile(x))
    dat.append(contains_time(x))
    dat.append(contains_ranking(x))
    dat.append(contains_numbers(x))
    dat.append(count_articles(x))
    dat.append(count_advjs(x))
    dat.append(count_comparative_superlative(x))
    return dat

def predict_score(subject_array):
  import pickle
  import warnings
  score = 0
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    filename = 'l1lr_clf_engfeats.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    prob = loaded_model.predict_proba(subject_array)
    score = prob[0][1]
    a= str(np.round(score,2))
  return a
#  return ("{0:.2f}".format(a))


def get_messages(subject_arr):
    msg = []
    if(subject_arr[0] == 0):
        msg.append("• include monetary values ($)")
    if(subject_arr[1] == 0):
        msg.append("• include parentheses, dashes, or ellipses ")
    if(subject_arr[2] == 0):
        msg.append("• include a question")
    if(subject_arr[3] == 1):
        msg.append("• remove the quotations ")
    if(subject_arr[4] == 1):
        msg.append("• remove copyright or trademark symbols ")
    if(subject_arr[8] == 1):
        msg.append("• remove '.com' information ")
    if(subject_arr[9] == 1):
        msg.append("• remove percentage values")
    if(subject_arr[14] == 0):
        msg.append("• use more adjectives or adverbs ")
    if(subject_arr[15] == 0):
        msg.append("• use more superlatives ")
    return msg
    




#Checks to see if the subject contains any mention of US Currency
def contains_currency(x):
    dols = '$' in x
    dol = 'dollar' in x
    cent = ' cent' in x
    usd = ' USD' in x
    return any([dols, dol, cent, usd])

#Determines if there are colons, paretheses, dashes, or ellipses
def contains_symbol_separators(raw_subject):

    symbol_separators = re.compile('|'.join([
        r'^.*(\:)\s.*$',    # colons need to have a space after
        r'^.*(\(.*\)).*$',  # parentheses.. includes things in parentheses not separated by spaces like al(oh)a
        r'^.*(\.{3,3}).*$', # ... must be three dots
        r'^.*\s(—)\s.*$',   # - must have spaces on either side
    ]))
    result = symbol_separators.match(raw_subject)
    
    return(result is not None)

#Determines if there's a question mark
def contains_question(raw_subject):
    return '?' in raw_subject

#Determines if there are double or single quotation marks
def contains_quotes(raw_subject):
    import re
    symbol_separators = re.compile('|'.join([
        r'(^|.+\s)(\'.+\').*$',  #single quotes: requires a space before first quotation mark
        r'(^|.+\s)(\".+\").*$',  #double quotes: requires a space before first quotation mark
    ]))
    result = symbol_separators.match(raw_subject)
    return(result is not None)

#Determines if the subject contains copyright, trademark, etc symbols
def contains_weird_symbols(raw_subject):
    r = '®' in raw_subject
    c = '©' in raw_subject
    tm = '™' in raw_subject   
    return any([r, c, tm])

#Counts the number of words
def find_num_words(raw_subject):
    return len(raw_subject.split())

#Determines the longest word in length
def find_longest_word_length(raw_subject):
    from itertools import groupby as gb

    words = raw_subject.split() 
    lwords = list(next(gb(sorted(words, key=len, reverse=True), key=len))[1])
    return len(lwords[0])

#Determines the length of the subject in total
def find_charlength(raw_subject):
    return len(raw_subject)

#Determines if '.com' is included in the subject
def contains_dotcom(raw_subject):
    return '.com' in raw_subject

#Determines if the subject contains numbers
def contains_numbers(raw_subject):
    return bool(re.search(r'\d', raw_subject))


def count_advjs(raw_subject):
    #JJ: adjective #RB: adverb
    adj_count = 0
    for tag in nltk.pos_tag(raw_subject.split()):
        if(('JJ'==tag[1]) or ('RB'==tag[1])):
            adj_count += 1
    return adj_count

def count_comparative_superlative(raw_subject):
    #JJR: adjective comparative.. better #RBR: adverb comparative
    #JJS: adjective superlative.. better #RBS: adverb superlative
    adj_count = 0
    for tag in nltk.pos_tag(raw_subject.split()):
        if(('JJR'==tag[1]) or ('RBR'==tag[1]) or 
            ('JJS'==tag[1]) or ('RBS'==tag[1])):
            adj_count += 1
    return adj_count

def count_articles(raw_subject):
    #DET
    art_count = 0
    for tag in nltk.pos_tag(raw_subject.split()):
        if('DET'==tag[1]):
            art_count += 1
    return art_count

#Determines if the subject contains a percentile
def contains_percentile(raw_subject):
    p = '%' in raw_subject
    l = 'percentile' in raw_subject
    t = 'percent' in raw_subject   
    return any([p, l, t])

#Checks if the subject contains a time element
def contains_time(raw_subject):
    mo = 'month' in raw_subject
    da = 'day' in raw_subject
    wk = 'week' in raw_subject   
    hr = 'hour' in raw_subject
    se = 'second' in raw_subject
    ti = 'time' in raw_subject   
    yr = 'year' in raw_subject
    de = 'decade' in raw_subject
    ce = 'century' in raw_subject 
    
    import re
    symbol_separators = re.compile('|'.join([
        r'(^|.+\s)([1-3][0-9]{3})[:\?!\.]*(?:\s+|$)',        #4 digit    
    ]))
    result = symbol_separators.match(raw_subject.lower())
    year = (result is not None)
     
    return any([mo, da, wk, hr, se, ti, yr, de, ce, year])

def contains_ranking(raw_subject):
    symbol_separators = re.compile('|'.join([
        r'(^|.+\s)\d+(th)[:\?!\.]*(?:\s+|$)',      #th
        r'(^|.+\s)\d+(st)[:\?!\.]*(?:\s+|$)',      #st
        r'(^|.+\s)\d+(nd)[:\?!\.]*(?:\s+|$)',      #nd
        r'(^|.+\s)\d+(rd)[:\?!\.]*(?:\s+|$)',      #th
        r'(^|.+\s)(ranked)[:\?!\.]*(?:\s+|$)',      #ranked
        r'(^|.+\s)(ranking)[:\?!\.]*(?:\s+|$)',     #ranking
        r'(^|.+\s)(rankings)[:\?!\.]*(?:\s+|$)',    #rankings
        r'(^|.+\s)(rank)[:\?!\.]*(?:\s+|$)',      #rank 
        r'(^|.+\s)(ranks)[:\?!\.]*(?:\s+|$)',      #ranks
       
        
    ]))
    result = symbol_separators.match(raw_subject.lower())
    return(result is not None)

