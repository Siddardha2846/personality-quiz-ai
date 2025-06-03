from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

app = Flask(__name__)

MODEL_FILENAME = "personality_model.pkl"
ENCODER_FILENAME = "label_encoder.pkl"
map_ans = {'A': 0, 'B': 1, 'C': 2}

def train_and_save_model():
    df = pd.read_csv("data.csv")
    X = df[['Q1', 'Q2', 'Q3']].apply(lambda col: col.map(map_ans))
    le = LabelEncoder()
    y = le.fit_transform(df['Label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(le, ENCODER_FILENAME)

    return model, le, model.score(X_test, y_test) * 100

def load_model_and_encoder():
    if os.path.exists(MODEL_FILENAME) and os.path.exists(ENCODER_FILENAME):
        model = joblib.load(MODEL_FILENAME)
        le = joblib.load(ENCODER_FILENAME)
        return model, le, None
    else:
        return train_and_save_model()

model, le, accuracy = load_model_and_encoder()

@app.route("/", methods=["GET", "POST"])
def index():
    personality = None
    if request.method == "POST":
        q1 = request.form.get("q1")
        q2 = request.form.get("q2")
        q3 = request.form.get("q3")

        df = pd.DataFrame([[q1, q2, q3]], columns=['Q1', 'Q2', 'Q3'])
        df = df.apply(lambda col: col.map(map_ans))
        pred = model.predict(df)[0]
        personality = le.inverse_transform([pred])[0]

    return render_template("index.html", personality=personality, accuracy=accuracy)

if __name__ == "__main__":
    app.run(debug=True)
