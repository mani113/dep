import pandas as pd
import numpy as np 
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')
os.getcwd()
os.chdir("E:\\ai")
df=pd.read_csv('winequality-red.csv' ,sep=';')

df.head()

x=df.drop(columns=['quality'])
y=df['quality']   
from imblearn.over_sampling import SMOTE
oversample=SMOTE(k_neighbors=4)
x,y=oversample.fit_resample(x,y)
from sklearn.model_selection import cross_val_score,train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=123)

    from sklearn.ensemble import ExtraTreesClassifier
    model=ExtraTreesClassifier()
model.fit(x,y)
pickle.dump(model,open('model1.pkl','wb'))
model1=pickle.load(open('model1.pkl','rb'))
print(model1.predict([[0,7.4,0.70,0.00,1.9,0.076,11.0,34.0,0.9978,3.51,0.56]]))

