from flask import Flask, request, render_template
import joblib
import numpy as np
import tensorflow as tf
import keras

# load model and scaler
try:
    model = tf.keras.models.load_model("lstm_model.h5")
    scaler = joblib.load("scaler.pkl")
    print("model and scaler imported.")
except Exception as e:
    print(f"{e}")

app = Flask(__name__)

# home route
@app.route("/")
def show():
    return render_template('index.html')

# after clicking the predict button
@app.route("/predict", methods=['POST'])
def pred():
    try:
        # 2. Get data from form
        # We expect a list of features in the correct order
        data = [float(x) for x in request.form.values()]
        features = np.array(data).reshape(1, -1)
        
        # 3. Scale the input
        scaled_features = scaler.transform(features)
        
        # 4. Reshape for LSTM: (Samples, Time_Steps, Features)
        # For a single snapshot demo, we repeat the row 50 times to create a sequence
        # In production, you would feed the actual last 50 cycles history.
        input_sequence = np.repeat(scaled_features, 50, axis=0).reshape(1, 50, -1)
        
        # 5. Predict
        prediction = model.predict(input_sequence)
        result = int(np.round(prediction[0][0]))
        
        # Clip result to 0 if it predicts negative life
        result = max(0, result)

        return render_template('index.html', prediction_text=f'Estimated Remaining Useful Life: {result} Cycles')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)