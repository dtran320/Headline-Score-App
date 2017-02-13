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

#Set the path to use NLTK stopwords
nltk.data.path.append('./nltk_data/')

#Returns True if raw_subject contains any mention of US Currency
def contains_currency(raw_subject):
    dols = '$' in raw_subject
    dol = 'dollar' in raw_subject
    cent = ' cent' in raw_subject
    usd = ' USD' in raw_subject
    return any([dols, dol, cent, usd])
    
#Returns True if there are colons, paretheses, dashes, or ellipses in raw_subject
def contains_symbol_separators(raw_subject):

    symbol_separators = re.compile('|'.join([
        r'^.*(\:)\s.*$',    # colons need to have a space after
        r'^.*(\(.*\)).*$',  # parentheses..
        r'^.*(\.{3,3}).*$', # ... must be three dots
        r'^.*\s(—)\s.*$',   # - must have spaces on either side
    ]))
    result = symbol_separators.match(raw_subject)
    return(result is not None)


#Returns True if there's a question mark in raw_subject
def contains_question(raw_subject):
    return '?' in raw_subject

#Returns True if there are double or single quotation marks in raw_subject
def contains_quotes(raw_subject):
    
    symbol_separators = re.compile('|'.join([
        r'(^|.+\s)(\'.+\').*$',  #single quotes: requires a space before first quotation mark
        r'(^|.+\s)(\".+\").*$',  #double quotes: requires a space before first quotation mark
    ]))
    result = symbol_separators.match(raw_subject)
    return(result is not None)

#Returns True if raw_subject contains copyright, trademark, etc symbols
def contains_weird_symbols(raw_subject):
    r = '®' in raw_subject
    c = '©' in raw_subject
    tm = '™' in raw_subject   
    return any([r, c, tm])

#Returns the number of words in raw_subject
def find_num_words(raw_subject):
    return len(raw_subject.split())

#Returns the length of the longest word in raw_subject
def find_longest_word_length(raw_subject):
    words = raw_subject.split() 
    lwords = list(next(gb(sorted(words, key=len, reverse=True), key=len))[1])
    return len(lwords[0])

#Returns the number of characters in the raw_subject
def find_charlength(raw_subject):
    return len(raw_subject)

#Returns True if '.com' is included in the raw_subject
def contains_dotcom(raw_subject):
    return '.com' in raw_subject

#Returns True if the raw_subject contains numbers
def contains_numbers(raw_subject):
    return bool(re.search(r'\d', raw_subject))


#Returns the number of adjectives and adverbs in raw_subject
def count_advjs(raw_subject):
    #JJ: adjective 
    #RB: adverb
    adj_count = 0
    for tag in nltk.pos_tag(raw_subject.split()):
        if(('JJ'==tag[1]) or ('RB'==tag[1])):
            adj_count += 1
    return adj_count

#Returns the number of comparative and superlative words in raw_subject
def count_comparative_superlative(raw_subject):
    #JJR: adjective comparative  
    #RBR: adverb comparative
    #JJS: adjective superlative  
    #RBS: adverb superlative
    adj_count = 0
    for tag in nltk.pos_tag(raw_subject.split()):
        if(('JJR'==tag[1]) or ('RBR'==tag[1]) or 
            ('JJS'==tag[1]) or ('RBS'==tag[1])):
            adj_count += 1
    return adj_count

#Returns the number of articles in raw_subject
def count_articles(raw_subject):
    #DET
    art_count = 0
    for tag in nltk.pos_tag(raw_subject.split()):
        if('DET'==tag[1]):
            art_count += 1
    return art_count

#Returns True if raw_subject contains a percentile
def contains_percentile(raw_subject):
    p = '%' in raw_subject
    l = 'percentile' in raw_subject
    t = 'percent' in raw_subject   
    return any([p, l, t])

#Returns True if raw_subject contains a time element
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
        r'(^|.+\s)([1-3][0-9]{3})[:\?!\.]*(?:\s+|$)', #4 digit    
    ]))
    result = symbol_separators.match(raw_subject.lower())
    year = (result is not None)
     
    return any([mo, da, wk, hr, se, ti, yr, de, ce, year])

#Returns True if raw_subject contains ranking information
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

#Returns the number of acronyms in raw_subject
def count_acronyms(raw_subject):
    regex = r"\b[A-Z][a-zA-Z\.]*[A-Z]\b\.?"
    return len(re.findall(regex, raw_subject))

#Returns True if raw_subject is all capitalized letters
def is_all_caps(raw_subject):
    num = sum(1 for c in raw_subject if c.isupper())
    return(num == len(raw_subject.replace(" ","")))

#Returns the number of capitalized letters in raw_subject
def number_of_caps(raw_subject):
    return sum(1 for c in raw_subject if c.isupper())

#Returns True if raw_subject begins with who, what, where, when, why, or how
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


# Takes raw_subject as input, and returns a string 
# where the words are converted to lower-case, 
# non-letters and stopwords are removed,  
# and then lemmatized using the wordnet lemmatizer
def clean_subject( raw_subject ):
    
    #Remove any numbers or letters in combination with numbers
    subject = ' '.join(s for s in raw_subject.split() if not any(c.isdigit() for c in s))

    # Convert to lower-case and remove non-letters   
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(subject.lower()) 
   
    # Remove stop words.  Convert the stop words to a set first for faster search
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]   
    
    # Lemmatize the words
    lmtzr = WordNetLemmatizer()
    lem_words = [lmtzr.lemmatize(w) for w in meaningful_words]   
    
    # Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( lem_words ))  


# Function to return a single string containing all of the adjectives, adverbs, and verbs
# from raw_subject using Stanford's POS Tagger
def return_aavs(raw_subject):
    jar = 'classifier_tools/stanford-postagger-2014-08-27/stanford-postagger.jar'
    model = 'classifier_tools/stanford-postagger-2014-08-27/models/english-left3words-distsim.tagger'
    
    #If using java 1.8, then you can use the more recent version of Stanford's POS Tagger
    #heroku buildpacks:set https://github.com/heroku/heroku-buildpack-jvm-common.git
    #jar = 'classifier_tools/stanford-postagger-2016-10-31/stanford-postagger.jar'
    #model = 'classifier_tools/stanford-postagger-2016-10-31/models/english-left3words-distsim.tagger'
    
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

    #Remove extra stop words
    extra_stops = set(['apple','android','usb'])
    meaningful_words2 = [w for w in meaningful_words if not w in extra_stops] 
    
    #Lemmatize the words
    lmtzr = WordNetLemmatizer()
    lem_words = [lmtzr.lemmatize(w) for w in meaningful_words2]   
    
    # Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( lem_words ))  


# Return a pandas dataframe of values for the 18 engineered features from the 
# model for raw_subject
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


# Returns the predicted probability of being a positive case (headline from VentureBeat)
# of raw_subject
def predict_proba_boaav_eng(raw_subject):
    
    #Get all of the adjectives, adverbs, and verbs
    aavs = return_aavs(raw_subject)

    #Clean, lemmatize, and remove stop and extra stop words
    subject = extra_clean_subject(aavs)

    #Use the count vectorizer to count the number of adjectives, adverbs, and verbs
    #that were identified to be important by the classifier model
    cvect_boaav = joblib.load('classifier_tools/cvect_stfd_adj_adv_verbs_n1000_ngram(1,1).pkl')  
    boaav_feats = cvect_boaav.transform([subject])
    vocab = cvect_boaav.get_feature_names()
    
    #Create a DataFrame with the Bag of Adjective, Adverb, and Verb word counts
    boaav_df = pd.DataFrame(boaav_feats.toarray())
    boaav_df.columns = vocab

    #Get a DataFrame with the values for the 18 engineered features from the model
    engfeats_df = get_eng_feats_df(raw_subject)
    
    #Combine the Bag of adverbs, adjectives, and verbs with the 18 engineered features
    df_combined_feats = pd.concat([engfeats_df, boaav_df], axis=1)

    #Apply the model on the combined feature values of raw_subject to get the predicted probability score
    comb_model = pd.read_pickle('classifier_tools/combined_boaav_engfeats_mod.pkl') 
    score = comb_model.predict_proba(df_combined_feats)[0][1]

    #Return the score, the engineered feature values, and the string of adjectives, adverbs, verbs
    return score, engfeats_df, aavs


# This function returns an array of templated sentences that suggest to remove 
# any words that were features with negative coefficients in the model.  
def get_suggestions_to_remove_words(subject):
    low_score_words = ['date', 'productive','emphasizes','reliable', 'promising',
                'coach','meeting','healthy','proudly','significant','innovative',
                'sustainable','enjoys','affordable','estimated','recognizes', 
                'maintains','published','honored','participate','offered',      
                'thrive','stand','charitable','outstanding','supporting',   
                'enhance','effective','rapidly','flexible','saving','recent',       
                'stress','worth','host','schedule' ,'serf' ,'special',      
                'electrical','endorses','achieves' ,'training','renowned',   
                'unique','attends','absolute','released','provides','featured',
                'likely','clean','process','local','global','newly','simplifies',
                'exciting','celebrates','certified','minimizing','qualified','anticipates',
                'relentless','complete','award','senior','profit','beautiful',
                'successfully','celebrating','recently','providing','customized',
                'higher','pack','pure','creative','fresh','thrives','awarded',
                'recognized','donates','important','reach','achieve','participates',
                'offering','true','book','publishes','fully','multiple','present','triple',
                'forward','prioritizes','named','living','strengthens','complimentary',
                'direct','current','receive','trusted','attend','ever','late','prestigious',
                'grow','approved','empower','finish','rental','remains','away',
                'receives','instead','best','additional','ensure','lower','high',
                'limitless','introduce','efficient','inspired','large','eliminate',
                'enhances','spending','enjoy','emerges','traditional','set','little',
                'increasing','print','prepares','protect','private','earns','shape',
                'successful','rising','alternative','listed','highly','breaking','giving',
                'quarterly','active','influential','offer','shipping','donate','introducing',
                'completely','poised'
               ]
    query = subject.lower()               
    msg = []
    for word in low_score_words:
        if word in query:
            message = "• Consider replacing the word '"+word+"'"
            msg.append(message)
    return msg 


# Get a list of suggestions for improvement given an array of the 
# engineered feature values. 
def get_messages(engfeats_arr):
    msg = []
    
    #Contains Quotes: Negative Coefficient Feature
    if(engfeats_arr[3] == 1):
        msg.append("• Remove the quotations ")

    #Starts with Who, What, Where, When, Why, How:  Positive Coefficient Feature
    if(engfeats_arr[16] == 0):
        msg.append("• Begin with Who, What, Where, When, Why, or How ")

    #Contains Currency: Positive Coefficient Feature
    if(engfeats_arr[0] == 0):
        msg.append("• Include monetary values ($)")

    #Contains copyright or trademark symbols: Negative Coefficient Feature
    if(engfeats_arr[4] == 1):
        msg.append("• Remove copyright or trademark symbols ")

    #Contains parentheses, dashes, ellipses: Positive Coefficient Features
    if(engfeats_arr[1] == 0):
        msg.append("• Include parentheses, dashes, or ellipses ")

    #If no adjectives or adverbs, suggest to use one: Positive Coefficient Feature
    if(engfeats_arr[12] == 0):
        msg.append("• Use more adjectives or adverbs ")

    #If no comparative or superlative words, suggest to use one: Positive Coefficient Fature
    if(engfeats_arr[13] == 0):
        msg.append("• Use more superlatives ")

    #If contains '.com', remove it: Negative Coefficient
    if(engfeats_arr[7] == 1):
        msg.append("• Remove '.com' information ")

    #If contains ranking information, remove it: Negative Coefficient Feature
    if(engfeats_arr[10] == 1):
        msg.append("• Remove ranking information")

    #If does not contain percentage information, include it: Positive Coefficient Feature
    if(engfeats_arr[8] == 0):
        msg.append("• Include percentage values")    
    #If all words are capitalized, suggest to downcase them: Negative Coefficient Feature
    if(engfeats_arr[15] == 1):
        msg.append("• Do not make all words capitalized ")
     
    return msg

