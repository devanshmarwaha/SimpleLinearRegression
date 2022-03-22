#importing required libraries
import numpy as np
import matplotlib.pyplot as mtp  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import accuracy_score,r2_score
import pickle


#Data preprocessing
data_set= pd.read_csv('Salary_Data.csv')  
x= data_set.iloc[:, 0:1].values  
y= data_set.iloc[:, 1].values  

#Model-Training & choosing
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3, random_state=0)  
regressor= LinearRegression()  
regressor.fit(x_train, y_train) 

#Predictions based on selected models , done by model
y_pred= regressor.predict(x_test)  
x_pred= regressor.predict(x_train)  


#Graphs
mtp.scatter(x_train, y_train, color="green")   
mtp.plot(x_train, x_pred, color="red")    
mtp.title("Salary vs Experience (Training Dataset)")  
mtp.xlabel("Years of Experience")  
mtp.ylabel("Salary(In Rupees)")  
mtp.show()   

mtp.scatter(x_test, y_test, color="blue")   
mtp.plot(x_train, x_pred, color="red")    
mtp.title("Salary vs Experience (Test Dataset)")  
mtp.xlabel("Years of Experience")  
mtp.ylabel("Salary(In Rupees)")  
mtp.show()  

pickle.dump(regressor,open('temp.pkl','wb'))
model=pickle.load(open('temp.pkl','rb'))
x1=float(input("Enter your Experience: "))
exp=np.array([[x1]])
prediction = model.predict(exp)
print("Your Salary should be :",prediction[0])

#Acuuracy checks
print(accuracy_score(y_test, y_pred.round(),normalize=True))
print(r2_score(y_test,y_pred))


