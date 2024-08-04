import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img , img_to_array

app = Flask(__name__)

model = load_model('snake_model.h5')

model.make_predict_function()

def predict_label(img_path):
	i = load_img(img_path, target_size=(128,128))
	i = img_to_array(i)/255.0
	i = i.reshape(1,128,128,3)

	p = model.predict(i)
	text = []
	if p < 0.5:
		p[0][0] *= 1000
		txt = 'Non Venomous'
	else:
		p[0][0] *= 100
		txt = 'Venomous'
	pre = '%.2f'%p[0][0]
	return pre,txt


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	

		img.save(img_path)

		p1 = predict_label(img_path)


	return render_template("index.html", prediction = p1[0], TEXT = p1[1], img_path = img_path)


if __name__ =='__main__':
	app.run(port = 3000 , debug = True)