import flask
import pandas as pan
import numpy as np
from PIL import Image
import joblib
import math
import easygui
from keras.models import load_model

model_1 = load_model('./saved_models/model_1.h5')
model_2 = joblib.load('./saved_models/model_2.pkl')
model_3 = load_model('./saved_models/model_3.h5')

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET','POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        filepath = easygui.fileopenbox()
        latitude = flask.request.form['latitude']
        longitude = flask.request.form['longitude']
        coords = [[latitude,longitude]]
        df = pan.DataFrame(coords, columns=['lat','long'])
        # Get the predictions
        raw_image = Image.open(filepath)
        img = np.asarray(raw_image)
        img = img.astype('float32')
        img /= 255.
        img = np.expand_dims(img, axis=0)
        prediction_1 = model_1(img, training=False)
        prediction_2 = model_2.predict(df)
        prediction_2 = np.eye(4)[prediction_2[0]]
        df3 = np.array([prediction_2[0],prediction_2[1],prediction_2[2],prediction_2[3],prediction_1[0][0],prediction_1[0][1],prediction_1[0][2],prediction_1[0][3]])
        df3 = np.expand_dims(df3, axis=0)
        prediction_3 = model_3(df3, training=False)
        result = np.argmax(prediction_3)
        answer = ''
        # Get the answer
        if result == 0:
            answer = 'desert'
        elif result == 1:
            answer = 'forest'
        elif result == 2:
            answer = 'grassland'
        else:
            answer = 'tundra'
        return flask.render_template('main.html', input_parameters={'Latitude':latitude, 'Longitude':longitude, 'File':filepath}, prediction=answer)
            
            
if __name__ == '__main__':
    app.run()