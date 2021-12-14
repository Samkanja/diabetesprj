import numpy as np
import pickle

loaded_model = pickle.load(open('/home/kskanja/stuff/models/trained_model.sav', 'rb'))

input_data = (5,166,72,19,175,25.8,0.587,51)
modified = np.asarray(input_data)
reshaped_data = modified.reshape(1, -1)




predition = loaded_model.predict(reshaped_data)
print(predition)
if predition[0]==0:
    print('patient is not diabetic')
else:
     print('patient is diabetic')