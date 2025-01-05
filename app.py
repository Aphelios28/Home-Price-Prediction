from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load(open('D:\Study\MachineLearning\HomePricePredict\Best_Model.pkl', 'rb'))
@app.route('/')
def hello():
    return render_template('index.html')

# prediction function   
@app.route('/predict', methods = ['POST']) 
def predict(): 
    A = [float(x) for x in request.form.values()]
    print(A)
    model_probability = model.predict([A])
    print(A)
    print(model_probability)
    prediction = "%0.2f"%abs(model_probability)
    return render_template('index.html', result = prediction)

if __name__ == "__main__":
    app.run(debug=True)