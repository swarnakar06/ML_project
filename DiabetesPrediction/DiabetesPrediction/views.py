

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from django.shortcuts import render
def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    data = pd.read_csv(r"C:\Users\Asus\Downloads\diabetes.csv")
    X = data.drop("Outcome", axis=1)
    Y = data['Outcome']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, Y_train)

    val1 = float(request.GET.get('n1',0))
    val2 = float(request.GET.get('n2',0))
    val3 = float(request.GET.get('n3',0))
    val4 = float(request.GET.get('n4',0))
    val5 = float(request.GET.get('n5',0))
    val6 = float(request.GET.get('n6',0))
    val7 = float(request.GET.get('n7',0))
    val8 = float(request.GET.get('n8',0))

    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    result1 = ""
    if pred ==[1]:
        result1 = "positive"
    else:
        result1 = "Negative"


    return render(request, "predict.html",{"result2":result1})