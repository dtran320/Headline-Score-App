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
  return redirect(url_for('search_results', query=query))

@app.route('/search_results/<query>')
def search_results(query):
  # Get the score, the 
  score, engfeats, aavs = predict_proba_boaav_eng(query)
  msgs_eng_feats = get_messages(engfeats.values[0])
 
  subject_array = ""
  msgs_remove_words = get_words_to_remove(query)
  
  msgs= msgs_remove_words + msgs_eng_feats 
  #msgs= msgs_replace_words + msgs_remove_words + msgs_eng_feats 
  sym = ""
  output = {'symbols': sym}  
  return render_template("results.html", results=score, query=query, output=output, score=score, msgs=msgs)


   


# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
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