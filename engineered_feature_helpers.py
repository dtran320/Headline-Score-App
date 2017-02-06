import nltk
import re
from itertools import groupby as gb
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from nltk.tag import StanfordPOSTagger
from nltk import word_tokenize
import pandas as pd

#Check the results of tests
def test_results(fcn_name, tests):
    num_failed = 0
    for test, expected in tests:
        result = fcn_name(test) 
        if result == expected:
            status = 'OK'
            status += ' (%s)' % result
        else:
            num_failed+=1
            status = 'FAIL'
            status += ' (%s)' % result
        print(test + '\t\t\t' + status)
        
    print("____________")
    print("Num Tests Failed: ", num_failed)

#Checks to see if the subject contains any mention of US Currency
def contains_currency(x):
    dols = '$' in x
    dol = 'dollar' in x
    cent = ' cent' in x
    usd = ' USD' in x
    return any([dols, dol, cent, usd])
    
currency_tests = [
    ('10 cents ', True),
    ('percent', False),
    ('$100', True),
    ('raised 5 million dollars in', True),
    ('5 USD', True),
   ]

test_results(contains_currency, currency_tests)

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
    #if(result is not None):
    #    symbol = [x for x in result.groups() if x is not None].pop()   
    #    return(symbol)
    #else:
    #    return None 
    
symbol_tests = [
    ('asdfasdf:', False),
    ('asdf: asdfa', True),
    ('(blah)', True),
    ('a(blah)b', True),
    ('blah...', True),
    ('blah... ', True),
    ('blah ...', True),
    ('b—b', False),
    ('b — b', True),
   ]

test_results(contains_symbol_separators, symbol_tests)

#Determines if there's a question mark
def contains_question(raw_subject):
    return '?' in raw_subject

#Determines if there are double or single quotation marks
def contains_quotes(raw_subject):
    
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
#    if(result is not None):
#        symbol = [x for x in result.groups() if x is not None].pop()     
#        return(symbol)
#    else:
#        return None
     
ranking_tests = [
    ('is 4th in line', True),
    ('is 4th! Get in line', True),
    ('4th seed', True),
    ('july 4th', True),
    ('july 4th?', True),
    ('is 1st? In', True),
    ('is 1st?a In', False),
    ('going 1st to', True),
    ('2nd in line', True),
    ('is 33rd:', True),
    ('is ranked 35', True),
    ('ranked.', True),
    ('and franks', False),
    ('ranks', True),
    ('is rank the', True),
    ('ranking', True),
    ('pranking', False),
   ]

test_results(contains_ranking, ranking_tests)


def count_acronyms(raw_subject):
    regex = r"\b[A-Z][a-zA-Z\.]*[A-Z]\b\.?"
    return len(re.findall(regex, raw_subject))

def is_all_caps(raw_subject):
    num = sum(1 for c in raw_subject if c.isupper())
    return(num == len(raw_subject.replace(" ","")))

def number_of_caps(raw_subject):
    return sum(1 for c in raw_subject if c.isupper())

def starts_with_5WH(raw_subject):
    symbol_separators = re.compile('|'.join([
        r'^(who).+$',      # Who, whoever, etc
        r'^(what).+$',      # What, whatever, etc
        r'^(where).+$',      # Where
        r'^(when).+$',      # When
        r'^(why).+$',      # Why
        r'^(how).+$',      # How
       
    ]))
    result = symbol_separators.match(raw_subject.lower())
    return(result is not None)

# Takes a raw subject line as input, and returns a string 
# where the words are converted to lower-case, 
# non-letters and stopwords are removed,  
# and then lemmatized using the wordnet lemmatizer
def clean_subject( raw_subject ):
    
    # Function to convert a raw subject to a string of words
    # The input is a single string (a raw pitch subject), and 
    # the output is a single string (a preprocessed pitch subject)
  
    #Remove any numbers or letters in combination with numbers
    subject = ' '.join(s for s in raw_subject.split() if not any(c.isdigit() for c in s))

    # Convert to lower-case and remove non-letters   
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(subject.lower()) 
   
    # Remove stop words.  Convert the stop words to a set first for faster search
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]   
    
    lmtzr = WordNetLemmatizer()
    lem_words = [lmtzr.lemmatize(w) for w in meaningful_words]   
    
    # Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( lem_words ))  


def return_aavs(raw_subject):
    
    jar = 'stanford-postagger-2016-10-31/stanford-postagger.jar'
    model = 'stanford-postagger-2016-10-31/models/english-left3words-distsim.tagger'
    pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

    sents = []
    for tag_tuple in pos_tagger.tag([raw_subject]):
        ## JJ: adj, RB: adverb, VB: verb
        if (('JJ' in tag_tuple[1]) or ('RB' in tag_tuple[1]) or ('VB' in tag_tuple[1])):
            sents.append(tag_tuple[0])
    return(" ".join(sents))

# Takes a raw subject line as input, and returns a string 
# where the words are converted to lower-case, 
# non-letters and stopwords are removed,  
# and then lemmatized using the wordnet lemmatizer
def extra_clean_subject( raw_subject ):
    
    # Function to convert a raw subject to a string of words
    # The input is a single string (a raw pitch subject), and 
    # the output is a single string (a preprocessed pitch subject)
  
    #Remove any numbers or letters in combination with numbers
    subject = ' '.join(s for s in raw_subject.split() if not any(c.isdigit() for c in s))

    # Convert to lower-case and remove non-letters   
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(subject.lower()) 
   
    # Remove stop words.  Convert the stop words to a set first for faster search
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]   

    extra_stops = set(['apple','android','usb'])
    meaningful_words2 = [w for w in meaningful_words if not w in extra_stops] 
    
    lmtzr = WordNetLemmatizer()
    lem_words = [lmtzr.lemmatize(w) for w in meaningful_words2]   
    
    # Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( lem_words ))  




# Creates and returns a tuple containing a dataframe containing 
# word counts (columns) for each data point (row) 
# and the count vectorizer, after creating pickle files for both.
# A simple print out is also shown of the words identified with count
def create_BOW_features(data, suffix_name, max_feats = 500, ngrams_range=(1,2), stop_wds=None):
    print("Creating a bag of words of size ",max_feats)

    # Initialize the "CountVectorizer" object
    cv = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = stop_wds,   \
                             ngram_range = ngrams_range, \
                             max_features = max_feats) 
    
    # Fit the model and learns the vocabulary; 
    # Transforms our training data into feature vectors. 
    # The input to fit_transform should be a list ofstrings.
    trainfeats = cv.fit_transform(data)

    # Convert the result to an array
    trainfeats = trainfeats.toarray()

    # Sum up the counts of each vocabulary word
    vocab = cv.get_feature_names()
    dist = np.sum(trainfeats, axis=0)

    # Save count vectorizer
    joblib.dump(cv, 'cvect_'+suffix_name+'_n'+str(max_feats)+'_ngram'+str(ngrams_range).replace(' ','')+'.pkl') 

    # Save the feature dataframe
    df = pd.DataFrame(trainfeats)
    df.columns = vocab
    df.to_pickle('df_'+suffix_name+'_n'+str(max_feats)+'_ngram'+str(ngrams_range).replace(' ','')+'.pkl') 
    
    print("Created count vectorizer: cvect_"+suffix_name+'_n'+str(max_feats)+'_ngram'+str(ngrams_range).replace(' ','')+'.pkl') 
    print("Created features df: df_"+suffix_name+'_n'+str(max_feats)+'_ngram'+str(ngrams_range).replace(' ','')+'.pkl\n') 
    
    # For each, print the vocabulary word and the number of times it 
    # appears in the training set
    print("Printing out counts for each word identified")
    for tag, count in zip(vocab, dist):
        print(tag, count)
    
    return df, cv


def get_eng_feats_df(raw_subject):
  
    engfeats_df = pd.DataFrame(columns=["subject"], data=[[raw_subject]])

    engfeats_df['cont_currency'] = engfeats_df['subject'].apply(lambda x: contains_currency(x))
    engfeats_df['cont_sym_separators'] = engfeats_df['subject'].apply(lambda x: contains_symbol_separators(x))
    engfeats_df['cont_question_mark'] = engfeats_df['subject'].apply(lambda x: contains_question(x))
    engfeats_df['cont_quotes'] = engfeats_df['subject'].apply(lambda x: contains_quotes(x))
    engfeats_df['cont_weird_symbols'] = engfeats_df['subject'].apply(lambda x: contains_weird_symbols(x))
    engfeats_df['num_words'] = engfeats_df['subject'].apply(lambda x: find_num_words(x))
    engfeats_df['longest_word_length'] = engfeats_df['subject'].apply(lambda x: find_longest_word_length(x))
    engfeats_df['cont_dotcom'] = engfeats_df['subject'].apply(lambda x: contains_dotcom(x))
    engfeats_df['cont_percentile'] = engfeats_df['subject'].apply(lambda x: contains_percentile(x))
    engfeats_df['cont_time'] = engfeats_df['subject'].apply(lambda x: contains_time(x))
    engfeats_df['cont_ranking'] = engfeats_df['subject'].apply(lambda x: contains_ranking(x))
    engfeats_df['num_articles'] = engfeats_df['subject'].apply(lambda x: count_articles(x))
    engfeats_df['num_advjs'] = engfeats_df['subject'].apply(lambda x: count_advjs(x))
    engfeats_df['num_comps_supers'] = engfeats_df['subject'].apply(lambda x: count_comparative_superlative(x))
    engfeats_df['num_acronyms'] = engfeats_df['subject'].apply(lambda x: count_acronyms(x))
    engfeats_df['is_all_caps'] = engfeats_df['subject'].apply(lambda x: is_all_caps(x))
    engfeats_df['starts_with_5WH'] = engfeats_df['subject'].apply(lambda x: starts_with_5WH(x))
    del engfeats_df['subject']
    return engfeats_df

def predict_boaav_eng(raw_subject):
    
    aavs = return_aavs(raw_subject)
    subject = extra_clean_subject(aavs)
    comb_model = pd.read_pickle('combined_boaav_engfeats_mod.pkl') 
    cvect_boaav = joblib.load('cvect_stfd_adj_adv_verbs_n1000_ngram(1,1).pkl')  
    boaav_feats = cvect_boaav.transform([subject])
    vocab = cvect_boaav.get_feature_names()
    
    boaav_df = pd.DataFrame(boaav_feats.toarray())
    boaav_df.columns = vocab
    engfeats_df = get_eng_feats_df(raw_subject)
    
    df_combined_feats = pd.concat([engfeats_df, boaav_df], axis=1)
    return comb_model.predict(df_combined_feats)[0]

def predict_proba_boaav_eng(raw_subject):
    
    aavs = return_aavs(raw_subject)
    subject = extra_clean_subject(aavs)
    comb_model = pd.read_pickle('combined_boaav_engfeats_mod.pkl') 
    cvect_boaav = joblib.load('cvect_stfd_adj_adv_verbs_n1000_ngram(1,1).pkl')  
    boaav_feats = cvect_boaav.transform([subject])
    vocab = cvect_boaav.get_feature_names()
    
    boaav_df = pd.DataFrame(boaav_feats.toarray())
    boaav_df.columns = vocab
    engfeats_df = get_eng_feats_df(raw_subject)
    
    df_combined_feats = pd.concat([engfeats_df, boaav_df], axis=1)
    return comb_model.predict_proba(df_combined_feats)[0][1]

def get_score_str(predicted_probability_double):
    #p = predict_proba_boaav_eng(raw_subject)[0][1]*100
    p = predicted_probability_double*100
    return '{0:.{1}f}'.format(p, 1)



