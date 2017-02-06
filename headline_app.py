import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask import render_template
from random import randint
#from helpers import *
from engineered_feature_helpers import *

VERSION = '0.0.1'
app = Flask(__name__)
app.config.from_object('config')



@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
                           title='Home')


@app.route('/search', methods=['POST'])
def search():
  query = request.form['search_input']
  return redirect(url_for('search_results', query=query))

@app.route('/search_results/<query>')
def search_results(query):
  score = predict_proba_boaav_eng(query)

  subject_array = ""
  print("Score: ",score)
  
  msgs = ""
  sym = ""
  output = {'symbols': sym}  
  return render_template("results.html", results=score, query=query, output=output, score=score, msgs=msgs)




@app.route('/index2')
def index2():
    return render_template("index2.html",
                           title='Home')


@app.route('/search2', methods=['POST'])
def search2():
  query = request.form['search_input']
  return redirect(url_for('search_results2', query=query))

@app.route('/search_results2/<query>')
def search_results2(query):
  subject_array = make_array(query)
  score = get_score(subject_array)
  msgs = get_messages(subject_array)
  sym = contains_symbol_separators(query)
  output = {'symbols': sym}  
  return render_template("results2.html", results=score, query=query, output=output, score=score, msgs=msgs)




@app.route('/request')
@app.route('/request/<query>')
def req(query=None):
  

  print("query", query)
  if query == None:
    print("nothing here")
    return render_template("index2.html", title='Home')
  else:
    print("something here")
    subject_array = make_array(query)
    score = get_score(subject_array)
    msgs = get_messages(subject_array)
    sym = contains_symbol_separators(query)
    output = {'symbols': sym}  
    return render_template("index2.html", results=score, query=query, output=output, score=score, msgs=msgs)

 

   

@app.route('/search3', methods=['POST'])
def search3():
  query = request.form['search_input']
  print("Query in search3: ",query)
  return redirect(url_for('req', query=query))



# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/ajax')
def ajax():
    return render_template('ajax.html')

# Route that will process the AJAX request, sum up two
# integer numbers (defaulted to zero) and return the
# result as a proper JSON response (Content-Type, etc.)
@app.route('/_add_numbers')
def add_numbers():
   
    headline_query = request.args.get('headline_query', "", type=str)
    subject_array = make_array(headline_query)
    score = get_score(subject_array)
    score_str = str(score)
    return jsonify(result=score_str)




if __name__ == '__main__':
    """
    For if you want to run the flask development server
    directly
    """
    port = int(os.environ.get("PORT", 5001))
    host = os.environ.get("HOST", "localhost")
    app.run(debug=True, host=host, port=port)