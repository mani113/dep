import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'fixed acidity':2, 'volatile acidity':9, 'citric acid':6,'residual sugar':7,'chlorides':8,'free sulfur dioxide':11,'total sulfur dioxide':10,'density':1,'pH':11,'sulphates':0.56,'alcohol':9.4})

print(r.json())