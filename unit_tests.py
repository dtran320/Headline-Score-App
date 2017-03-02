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
from engineered_feature_helpers import *

run_code_tests = True

# Runs a quick test to see if a function (fcn_name)
# outputs what is expected.  Tests is an array of tuples,
# where the first item is the tuple is the input to fcn_name,
# and the second item in the tuple is the expected outcome
def test_results(fcn_name, tests):
    num_failed = 0
    print("_____________________\nTesting ",fcn_name)
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
        
    print("* ",num_failed, " Tests Failed *")
   
    return num_failed


if(run_code_tests):
    
    tot_failed = 0
    #Tests for contains_currency
    currency_tests = [
        ('10 cents ', True),
        ('percent', False),
        ('$100', True),
        ('raised 5 million dollars in', True),
        ('5 USD', True),
        ]
    tot_failed +=  test_results(contains_currency, currency_tests)
    
    #Tests for contains_symbol_separators
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
    tot_failed += test_results(contains_symbol_separators, symbol_tests)

    #Tests for contains_question
    question_tests = [
        ('Hello?', True),
        ('Hello?  There', True),
        ('?Hellow', True),
        ('raised $500', False)
       ]
    tot_failed += test_results(contains_question, question_tests)
    
    #Tests for contains_quotes
    quotes_tests = [
        ('"hi there"', True),
        ('" hi', False),
        ("'hi there'", True),
        ("' hello", False)
    ]
    tot_failed += test_results(contains_quotes, quotes_tests)
    
    #Tests for contains_weird_symbols
    symbols_tests = [
        ('® hi', True),
        ('hello ©', True),
        ('™ ', True),
        ("TM", False)
    ]
    tot_failed += test_results(contains_weird_symbols, symbols_tests)
    
    #Tests for find_num_words
    num_words_tests = [
        ('hello there', 2),
        ('hello', 1),
        ('', 0),
        ("hello there: how are you?", 5)
    ]
    tot_failed += test_results(find_num_words, num_words_tests)
    
    #Tests for find_longest_word_length
    longest_word_tests = [
        ('hello there', 5),
        ('abrcacadabra ', 12),
        ('', 0),
    ]
    tot_failed += test_results(find_longest_word_length,longest_word_tests)
    
    #Tests for find_charlength
    charlength_tests = [
        ('hello there', 11),
        ('abrcacadabra hello there how are you ', 37),
        ('', 0),
    ]
    tot_failed += test_results(find_charlength, charlength_tests)
    
    #Tests for contains_dotcom
    dotcom_tests = [
        ('hello.com',True),
        ('abrcacadabra hello there how are you ', False),
       
    ]
    tot_failed += test_results(contains_dotcom, dotcom_tests)
   
    #Tests for contains_numbers
    numbers_tests = [
        ('3K',True),
        ('abrcacadabra hello there how are you ', False),
        ('3 of us',True),
    ]
    tot_failed += test_results(contains_numbers, numbers_tests)

    # Tests for contains_ranking    
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
    tot_failed += test_results(contains_ranking, ranking_tests)
    
    # Tests for count_acronyms    
    acronyms_tests = [
        ('ABC', 1),
        ('is 4th! Get in line', 0),
        ('CEO of going 1st to', 1),
        ('Using their API', 1),
        ('CEO and CSO', 2),
       
    ]
    tot_failed += test_results(count_acronyms, acronyms_tests)
  
    # Tests for is_all_caps 
    caps_tests = [
        ('ABC', True),
        ('is 4th! Get in line', False),
        ('CEO IS THE BEST', True),
        ('Using their API', False),
        ('CEO and CSO', False),
       
    ]
    tot_failed += test_results(is_all_caps, caps_tests)

    # Tests for starts_with_5WH
    w5h_tests = [
        ('Who ', True),
        ('What is 4th! Get in line', True),
        ('Where is the CEO', True),
        ('When ', True),
        ('Why  am I ', True),
        ('How ', True),
        (' Who am I ', False),
        (' What is 4th! Get in line', False),
        (' Where is the CEO', False),
        (' Why', False),
        (' When is this', False),
        (' How is this', False),
    ]
    tot_failed += test_results(starts_with_5WH, w5h_tests)

    
    print("\n+---------------------------------------+")
    print(tot_failed," Total Tests Failed for all functions: ")