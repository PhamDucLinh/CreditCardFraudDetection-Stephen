# Flask API Demo for Credit Card Fraud Detection
# Endpoint /predict accepts JSON transaction -> fraud probability

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # TODO: Implement prediction logic
    pass

if __name__ == '__main__':
    app.run(debug=True)
