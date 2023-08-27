from flask import Flask,render_template,request,redirect,session
import tensorflow as tf
import os
from os.path import isfile, join
import numpy as np
import shutil
from tensorflow import keras
from pathlib import Path
from IPython.display import display, Audio
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model


import textblob            
from textblob import TextBlob
import speech_recognition as sr
import os
import random

from flask_mysqldb import MySQL
import MySQLdb.cursors
import re

app = Flask(__name__)

valid_split = 0.3   #training and testing data split 70:30

shuffle_seed = 43  

sample_rate = 16000

scale = 0.5

batch_size = 128 # 128 voices for training at once as a bag

epochs = 20 # number of passes a training dataset takes around an algorithm

valid_split = 0.3   #training and testing data split 70:30

shuffle_seed = 43  

sample_rate = 16000

scale = 0.5

batch_size = 128 # 128 voices for training at once as a bag

epochs = 20 # number of passes a training dataset takes around an algorithm




def add_noise(audio, noises=None, scale=0.5):
    if noises is not None:
        tf_rnd = tf.random.uniform(
            (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)

        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

        audio = audio + noise * prop * scale

    return audio


def path_to_audio(path):
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, sample_rate)
    return audio  

def paths_and_labels_to_dataset(audio_paths, labels):
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def audio_to_fft(audio):
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])

class_names = ['Aakash', 'Anagha', 'Atharva', 'Shrushti', 'Sneha', 'Tejas']

def predict(model,path, labels):
    test = paths_and_labels_to_dataset(path, labels)


    test = test.shuffle(buffer_size=batch_size * 8, seed=shuffle_seed).batch(
    batch_size
    )
    test = test.prefetch(tf.data.experimental.AUTOTUNE)


    # test = test.map(lambda x, y: (add_noise(x, noises, scale=scale), y))

    for audios, labels in test.take(1):
        ffts = audio_to_fft(audios)
        y_pred = model.predict(ffts)
        rnd = np.random.randint(0, 1, 1)
        audios = audios.numpy()[rnd, :]
        labels = labels.numpy()[rnd]
        y_pred = np.argmax(y_pred, axis=-1)[rnd]

    return class_names[y_pred[0]]

path = "D:/BENewAudio/marathiDataset/audio/"
path_list = os.listdir(path)
random_speaker = random.choice(path_list)
wave_path = path + random_speaker + "/"
wave_list = os.listdir(wave_path)
random_wave = random.choice(wave_list)
audio_path = wave_path + random_wave
filename = audio_path


@app.route("/dashboard")
def dashboard():
    model = load_model('model.h5')
    route = [filename]
    labels = ["unknown"]
    predicted = predict(model,route,labels)


    # Define a function to perform sentiment analysis on a given audio file
    def analyze_sentiment(filename):
        # Load the audio file and transcribe the speech to text
        r = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            audio = r.record(source)
        text = r.recognize_google(audio, language='mr-IN') # Specify Marathi language

        # Use TextBlob to perform sentiment analysis on the transcribed text
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity

        # Print the sentiment score and label
        if sentiment > 0:
            return ("Positive sentiment:", sentiment)
        elif sentiment < - 0.3:
            return ("Negative sentiment:", sentiment)
        else:
            return("Neutral sentiment")


    print(filename)


    sentimentAnalysis = analyze_sentiment(filename)
    return render_template('dashboard.html',predicted = predicted, route = route, sentimentAnalysis = sentimentAnalysis)




@app.route('/home')
def home():
    return render_template('home.html') 




app.secret_key = 'xyzsdfg'
  
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'userdb'
  
mysql = MySQL(app)
  

@app.route('/', methods =['GET', 'POST'])
def login():
    mesage = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s AND password = % s', (email, password, ))
        user = cursor.fetchone()
        if user:
            session['loggedin'] = True
            session['userid'] = user['userid']
            session['name'] = user['name']
            session['email'] = user['email']
            mesage = 'Logged in successfully !'
            return render_template('login.html', mesage = mesage)
        else:
            mesage = 'Please enter correct email / password !'
    return render_template('login.html', mesage = mesage)

           
@app.route('/register', methods =['GET', 'POST'])
def register():
    mesage = ''
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form :
        name = request.form['name']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s', (email, ))
        account = cursor.fetchone()
        if account:
            mesage = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            mesage = 'Invalid email address !'
        elif not userName or not password or not email:
            mesage = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO user VALUES (NULL, % s, % s, % s)', (name, email, password, ))
            mysql.connection.commit()
            mesage = 'You have successfully registered !'
    # elif request.method == 'POST':
    #     mesage = 'Please fill out the form !'
    return render_template('register.html', mesage = mesage)

@app.route('/userguide')
def userguide():
    return render_template('userguide.html')
 
# main driver function
if __name__ == '__main__':
    app.run(debug = True) 