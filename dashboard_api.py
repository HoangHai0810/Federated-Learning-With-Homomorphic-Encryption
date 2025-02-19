from flask import Flask, jsonify
from flask_cors import CORS
from threading import Thread

app = Flask(__name__)
CORS(app)
training_status = {
    "round": 0,
    "loss": 0.0,
    "accuracy": 0.0,
    "aggregation_time": 0.0,
    "avg_encryption_time": 0.0,
    "avg_decryption_time": 0.0,
    "log": []
}

@app.route("/status", methods=["GET"])
def get_status():
    return jsonify(training_status)

def start_dashboard():
    app.run(host="0.0.0.0", port=5000)

def run_dashboard_in_thread():
    thread = Thread(target=start_dashboard, daemon=True)
    thread.start()
