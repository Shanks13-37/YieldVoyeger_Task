from flask import Flask, render_template, request
from model import generate_answer

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", answer="", question="")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form["question"]
    answer = generate_answer(question)
    return render_template("index.html", answer=answer, question=question)

if __name__ == "__main__":
    app.run(debug=True)
