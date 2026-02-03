from flask import Flask, request, render_template
import random

app = Flask(__name__)

# home route
@app.route("/")
def show():
    return render_template('index.html')

# after clicking the predict button
@app.route("/predict", methods=['POST'])
def pred():
    num = random.randint(0,100)
    return render_template('index.html', prediction_text=f'Estimated Remaining Useful Life: {num} Cycles')

if __name__ == '__main__':
    app.run(debug=True)