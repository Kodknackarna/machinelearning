from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


app = Flask(__name__)

model_banana_quality = "finalized_model_BananaQuality.sav"
loaded_model = pickle.load(open(model_banana_quality, 'rb'))
rf_model = loaded_model['model']
model_banana_quality_accuracy = loaded_model['accuracy']

model_mobile_price = "finalized_model_mobile_price.sav"
loaded_mobile_model = pickle.load(open(model_mobile_price, 'rb'))
svc_model = loaded_mobile_model['model']
model_mobile_price_accuracy = loaded_mobile_model['accuracy']



@app.route('/PredictBanana', methods=['POST'])
def predict_banana():
    data = request.get_json(force=True) #extraherar json-datan från requesten

    prediction_data = pd.DataFrame([data]) #konventerar json-data till pandas dataframe

    prediction = rf_model.predict(prediction_data) #förutsägelsen, quality ska inte vara med i features eftersom d är target

    prediction_str = str(prediction[0]) #omvandlar svaret till string


    return jsonify({'prediction banana good or bad': prediction_str, 'accuracy med random forest': model_banana_quality_accuracy})


@app.route('/PredictMobile', methods=['POST'])
def predict_mobile():
    data = request.get_json(force=True) #extraherar json-datan från requesten

    prediction_data = pd.DataFrame([data]) #konventerar json-data till pandas dataframe

    prediction = svc_model.predict(prediction_data) #förutsägelsen, quality ska inte vara med i features eftersom d är target

    prediction_str = str(prediction[0]) #omvandlar svaret till string


    return jsonify({'prediction mobile price range': prediction_str, 'accuracy med svc': model_mobile_price_accuracy})

if __name__ == '__main__':
    app.run(debug=True)

''' good banana
{"Size": 0.02,
"Weight": 1,
"Sweetness": 0.5,
"Softness": 0.03,
"HarvestTime": -1,
"Ripeness": 0.1,
"Acidity": -0.2}
'''

''' bad banana
{"Size": 0.02,
"Weight": 2,
"Sweetness": -1.6,
"Softness": -0.8,
"HarvestTime": 2,
"Ripeness": -0.5,
"Acidity": -1}
'''

'''
{"battery_power": 1500,
"blue": 1,
"clock_speed": 2.0,
"dual_sim": 1,
"fc": 10,
"four_g": 1,
"int_memory": 40,
"m_dep": 0.8,
"mobile_wt": 175,
"n_cores": 6,
"pc": 15,
"px_heigth": 1200,
"px_width": 1000,
"ram": 3000,
"sc_h": 15,
"sc_w": 10,
"talk_time": 18,
"three_g": 1,
"touch_screen": 1,
"wifi": 1}
'''