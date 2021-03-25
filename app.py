import numpy as np
from flask import Flask, request, jsonify, render_template

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [str(x) for x in request.form.values()]

    cv = CountVectorizer()
    count_matrix = cv.fit_transform(int_features)

    # get the match percentage
    matchPercentage = cosine_similarity(count_matrix)[0][1] * 100
    matchPercentage = round(matchPercentage, 2) # round to two decimal
 
    return render_template('home.html', prediction_text="Your resume matches about {}% of the job description.".format(matchPercentage))


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host='0.0.0.0', port= 8080)