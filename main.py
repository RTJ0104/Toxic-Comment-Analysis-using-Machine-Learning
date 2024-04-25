import keras
import pickle
import re
import nltk
import string
from nltk.corpus import stopwords
stopword = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")

from tensorflow.keras.preprocessing import sequence
from flask import Flask, render_template, request, redirect, url_for
app = Flask(__name__)

load_model=keras.models.load_model("./model/hateModel.h5")
with open('./model/tokenizer.pickle', 'rb') as handle:
    load_tokenizer = pickle.load(handle)
    
result = ""

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
    

@app.route("/", methods=['GET', 'POST'])
def home():
    global result
    if request.method == 'POST':
        comment = request.form['comment']
        processed_comment = [clean_text(comment)]
        seq = load_tokenizer.texts_to_sequences(processed_comment)
        padded = sequence.pad_sequences(seq, maxlen=300)
        pred = load_model.predict(padded)
        if pred <= 0.5:
           result = "NO HATE SPEECH DETECTED"
        else:
           result = "HATE SPEECH DETECTED"
        return render_template("index.html", result = result)
    return render_template("index.html")
    
if __name__ == "__main__":
    app.run(debug=True)