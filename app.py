from flask import Flask, request, render_template, json
from numpy import loadtxt
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import re
import base64



app = Flask(__name__)

model = load_model('model.h5', compile = True)

@app.route('/')
def canvas():
    return render_template('mnist.html')

@app.route('/predict/', methods=['GET','POST'])
def predict():
    # get data from drawing canvas and save as image
    parseImage(request.get_data())


    img = load_img('canvas.png', grayscale = True, target_size= (28,28))

    img = img_to_array(img)

    img = img.reshape(1,28,28,1)

    img = img.astype('float32')

    img = img / 255.0

    out = model.predict(img)
    print(out)
    print(np.argmax(out, axis=1))
    response = np.array_str(np.argmax(out, axis=1))
    return response 
    
def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('canvas.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

if __name__ == '__main__':
    app.run(debug=True)