import os
from flask import Flask, render_template, request, redirect, url_for
from flask import render_template
from random import randint

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
  #db.posts.find( {$text: {$search :"Third"}})
  
  
  return render_template("results.html", query=query)



if __name__ == '__main__':
    """
    For if you want to run the flask development server
    directly
    """
    port = int(os.environ.get("PORT", 5001))
    host = os.environ.get("HOST", "localhost")
    app.run(debug=True, host=host, port=port)