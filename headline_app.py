import os
from flask import Flask, render_template, request, redirect, url_for
from random import randint
from engineered_feature_helpers import *


VERSION = '0.0.1'
app = Flask(__name__)
app.config.from_object('config')



@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html", title='Home')

@app.route('/search', methods=['POST'])
def search():
  #Remove any newline characters 
  query = request.form['search_input'].replace('\n', ' ')
  return redirect(url_for('score_results', query=query))

#Route for returning results for a query
@app.route('/score_results/<query>')
def score_results(query):

  # Get the score, the values for engineered features, 
  #and a string of adjectives, adverbs, and verbs from the query.
  score, engfeats, aavs = predict_proba_boaav_eng(query)

  #Get a list of suggestions for improvement based on values of engineered features
  msgs_eng_feats = get_messages(engfeats.values[0])
 
  #Get a list of suggestions to remove negative coefficient adjective, adverb, verb words
  remove_words_suggestions = get_suggestions_to_remove_words(aavs)
  
  #Append the two lists of suggestions
  all_msgs= remove_words_suggestions + msgs_eng_feats 
  
  return render_template("results.html", query=query, score=score, msgs=all_msgs)


   
# This route is for the about page with the blog description
@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    """
    For if you want to run the flask development server
    directly
    """
    port = int(os.environ.get("PORT", 5001))
    host = os.environ.get("HOST", "localhost")
    app.run(debug=True, host=host, port=port)