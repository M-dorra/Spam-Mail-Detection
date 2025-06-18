from flask import Flask, render_template, request
import pickle

with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app=Flask(__name__)

@app.route('/',methods=["GET","POST"])

def index():
    prediction=None
    if request.method=="POST":
        email_text=request.form["email"]
        data=vectorizer.transform([email_text])
        result=model.predict(data)[0]
        print(result)
        prediction="Spam" if result==1 else "Not Spam"
    return render_template("index.html",prediction=prediction)


if __name__=="__main__":
    app.run(debug=True)