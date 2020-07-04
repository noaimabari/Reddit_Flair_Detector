from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import praw
from praw.models import MoreComments
import pickle
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from nltk import pos_tag
import numpy as np


app = Flask(__name__) #creating a flask app
app.config['UPLOAD_FOLDER'] = os.getcwd() # path of the current working directory

model = pickle.load(open('model1.pkl', 'rb')) # loading the model
token_vec = pickle.load(open('transform1.pkl', 'rb')) # loading the tdifd vectorizer 

stop = stopwords.words("english")
punctuations = list(string.punctuation)
stop = stop + punctuations
lt = WordNetLemmatizer()

def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

## combining the above three functions 
## cleaning words
## removing stop words, punctuations, emojis
## performing lemmatization on the words

def clean_review(words):
    output_words = []
    for w in words:
        w = deEmojify(w) # removing emojis
        if w.lower() not in stop: # removing stopwords
            try:
                pos = pos_tag([w]) # finding the pos tag of the word
                clean_word = lt.lemmatize(w,pos = get_simple_pos(pos[0][1])) # lemmatizing the word
                output_words.append(clean_word.lower()) # adding the word in lower case as cleaned word
            except:
                continue
    return np.array(output_words) # returning the array of cleaned words

def get_cleaned_data(text):
    
    words = clean_review(text.split())
    
    cleaned_text = " ".join(words) ## joining the cleaned words 
    return cleaned_text

# function to preprocess text
def preprocess(text):
    text = text.replace("//", " ")
    text = text.replace('.', ' ')
    text = text.replace('https:', ' ')
    text = text.replace('_', ' ')
    text = text.replace('-', ' ')
    text = text.replace("/", " ")
    text = text.replace("'\'", " ")
    text = text.replace("'", " ")
    text = text.replace('[',  " ")
    text = text.replace('='," ")
    text = text.replace(']'," ")
    text = text.replace('['," ")
    text = text.replace(')', ' ')
    text = text.replace('(', ' ')
    text = text.replace('\\n', ' ')
    text = text.replace('\\t', ' ')
    text = text.replace('\\', ' ')
    text = text.replace('@', ' ')  
    text = text.replace('<', ' ') 
    text = text.replace('>', ' ') 
    text = text.replace("'", ' ') 
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ["Post"])
def predict():
    reddit = praw.Reddit(client_id='#',
                     client_secret='#',
                     user_agent='me')
    if request.method == 'POST':
        url = request.form['url']
        submission = reddit.submission(url=url)
        title_text = submission.title
        comments = ''
        for top_level_comment in submission.comments:
            if isinstance(top_level_comment, MoreComments):
                continue
            comments = comments + top_level_comment.body
        data = preprocess(comments + title_text)
        data = get_cleaned_data(data)
        data = token_vec.transform([data]).toarray()
        flair = model.predict(data)[0]
   
    return render_template('index.html', prediction_text = "Flair of the given reddit post is : {} " .format(flair))


@app.route('/automated_testing', methods = ['POST'])
def automated_testing():
    predictions = {}
    if request.method == 'POST':
        f = request.files['upload_file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        reddit = praw.Reddit(client_id='#', client_secret='#', user_agent='me')
        with open(f.filename, 'r') as file_obj:
            file_data = file_obj.readlines()
            for row in file_data:
                submission = reddit.submission(url=row) # each line in the file is a link to the post
                title_text = submission.title 
                comments = ''
                for top_level_comment in submission.comments:
                    if isinstance(top_level_comment, MoreComments):
                        continue
                    comments = comments + top_level_comment.body
                data = preprocess(comments + title_text) # preprocessing data
                data = get_cleaned_data(data)
                data = token_vec.transform([data]).toarray() # converting it into a sparse matrix using tf idf vectorizer
                flair = model.predict(data) 
                predictions[row] = str(flair[0])
        return jsonify(predictions) 

if __name__ == '__main__':
    app.run(debug=True)
     