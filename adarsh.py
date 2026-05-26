
import pandas as pd
data=pd.read_csv("machine_failure.csv")
#print(f"so the data is:\n{data}")

from sklearn.preprocessing import LabelEncoder  
label_encoder = LabelEncoder()
data['Type_encoded'] = label_encoder.fit_transform(data['Type'])
#print(f"after encoding the data is:\n{data}")
x=data[['Type_encoded', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF' ]]
y=data[['Machine failure']]

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()  
model.fit(x,y)
print('coef:',model.coef_)
print('intercept:',model.intercept_)
type = int(input('enter the type of machine (L, M, H):0,1,2: '))
air_temp = float(input('enter the air temperature in K: ')) 
process_temp = float(input('enter the process temperature in K: '))
rotational_speed = float(input('enter the rotational speed in rpm: '))      
torque = float(input('enter the torque in Nm: '))
tool_wear = float(input('enter the tool wear in min: '))
twf = int(input('enter the TWF (0 or 1): '))
hdf = int(input('enter the HDF (0 or 1): '))
pwf = int(input('enter the PWF (0 or 1): '))
osf = int(input('enter the OSF (0 or 1): '))
rnf = int(input('enter the RNF (0 or 1): '))
print(model.predict([[type, air_temp, process_temp, rotational_speed, torque, tool_wear, twf, hdf, pwf, osf, rnf]]))  



