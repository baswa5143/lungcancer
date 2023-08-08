import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
app=Flask(__name__)
model2=pkl.load(open('cancer.pkl','rb'))
@app.route('/')
def index():
  return render_template("index.html")
@app.route('/predict',methods=["POST"])
def predict():
      
      
      
      if request.method == 'POST':
            
            gender=int(request.form["gender"])
            
            age=int(request.form["age"])
            smoking=int(request.form["smoking"])
            
            yellow_fingers=int(request.form["yellow_fingers"])
            
            anxiety=int(request.form["anxiety"])
           
            peer_pressure=int(request.form["peer_pressure"])
            
            Chronic_disease=int(request.form["Chronic_disease"])
            
            fatigue=int(request.form["fatigue"])
            
            Allergy=int(request.form["Allergy"])
            
            wheezing=int(request.form["wheezing"])
            
            Alcohol=int(request.form["Alcohol"])
            
            Coughing=int(request.form["Coughing"])
            
            shortness_of_breath=int(request.form["shortness_of_breath"])
            
            swallowing_difficulty=int(request.form["swallowing_difficulty"])
            
            Chest_pain=int(request.form["Chest_pain"])
            
            final_features=np.array([[gender,age,smoking,yellow_fingers,anxiety,peer_pressure,Chronic_disease,fatigue,Allergy,wheezing,Alcohol,Coughing,shortness_of_breath,swallowing_difficulty,Chest_pain]])
            prediction=model2.predict(final_features)
      return render_template("result.html",predictions= prediction)
if __name__=="__main__":
  app.run(debug=True)