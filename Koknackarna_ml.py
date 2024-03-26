from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import pickle
import pandas as pd

app = Flask(__name__)

app.config['JWT_SECRET_KEY'] = 'kodknackarna' #vår hemliga nyckel, används för signering och bör vara slumpad textsträng, och egentligen inte hårdkodad
jwt = JWTManager(app)

model_banana_quality = "finalized_model_BananaQuality.sav"
loaded_banana_model = pickle.load(open(model_banana_quality, 'rb'))
rf_model = loaded_banana_model['model']
model_banana_quality_accuracy = loaded_banana_model['accuracy']

model_mobile_price = "finalized_model_mobile_price.sav"
loaded_mobile_model = pickle.load(open(model_mobile_price, 'rb'))
svc_model = loaded_mobile_model['model']
model_mobile_price_accuracy = loaded_mobile_model['accuracy']


@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username', None)
    password = request.json.get('password', None)

    # kontrollerar användarnamn och lösenord är rätt
    if username != 'yves' or password != '123':
        return jsonify({"msg": "Bad username or password"}), 401

    # Skapar ett nytt token för användaren
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)

@app.route('/PredictBanana', methods=['POST'])
@jwt_required()
def predict_banana():
    current_user = get_jwt_identity()   # identiferar med jwt token

    data = request.get_json(force=True) #extraherar json-datan från requesten

    prediction_data = pd.DataFrame([data]) #konventerar json-data till pandas dataframe

    prediction = rf_model.predict(prediction_data) #förutsägelsen, quality ska inte vara med i features eftersom d är target

    prediction_str = str(prediction[0]) #omvandlar svaret till string


    return jsonify({'prediction banana good or bad': prediction_str, 'accuracy med random forest': model_banana_quality_accuracy})


@app.route('/PredictMobile', methods=['POST'])
@jwt_required()
def predict_mobile():
    current_user = get_jwt_identity()  # identiferar med jwt token

    data = request.get_json(force=True) #extraherar json-datan från requesten

    prediction_data = pd.DataFrame([data]) #konventerar json-data till pandas dataframe

    prediction = svc_model.predict(prediction_data) #förutsägelsen, quality ska inte vara med i features eftersom d är target

    prediction_str = str(prediction[0]) #omvandlar svaret till string


    return jsonify({'prediction mobile price range': prediction_str, 'accuracy med svc': model_mobile_price_accuracy})

if __name__ == '__main__':
    app.run(debug=True)


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