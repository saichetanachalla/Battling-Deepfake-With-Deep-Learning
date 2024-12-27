

import joblib
import numpy as np;
import pandas as pd;
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb
import matplotlib.pyplot  as plt;
from sklearn.model_selection  import train_test_split
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle
gmail_list=[]
password_list=[]
gmail_list1=[]
password_list1=[]
import numpy as np;
import pandas as pd;
import matplotlib.pyplot  as plt;
from sklearn.model_selection  import train_test_split
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle
import random



import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np;
import pandas as pd;
import matplotlib.pyplot  as plt;
from sklearn.model_selection  import train_test_split
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle


from yield_code import yield_fn


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# Load column names from the pickle file
with open('column_names2.pkl', 'rb') as file:
    loaded_column_names = pickle.load(file)

# Print the loaded column names
print(loaded_column_names)


disease_dic= ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___healthy',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

from model_predict  import pred_leaf_disease


features_list=loaded_column_names
# Load the list from the pickle file
with open('zero_list2.pkl', 'rb') as file:
    loaded_list = pickle.load(file)

# Print the loaded list
print(loaded_list)

features_list1=loaded_list
from recamandation_code import recondation_fn


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('login44.html') 

@app.route('/logedin',methods=['POST'])
def logedin():
    
    int_features3 = [str(x) for x in request.form.values()]
    print(int_features3)
    logu=int_features3[0]
    passw=int_features3[1]
   # if int_features2[0]==12345 and int_features2[1]==12345:

    import MySQLdb


# Open database connection
    db = MySQLdb.connect("localhost","root","","ddbb" )

# prepare a cursor object using cursor() method
    cursor = db.cursor()
    cursor.execute("SELECT user FROM user_register")
    result1=cursor.fetchall()
              #print(result1)
              #print(gmail1)
    for row1 in result1:
                      print(row1)
                      print(row1[0])
                      gmail_list.append(str(row1[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(gmail_list)
    

    cursor1= db.cursor()
    cursor1.execute("SELECT password FROM user_register")
    result2=cursor1.fetchall()
              #print(result1)
              #print(gmail1)
    for row2 in result2:
                      print(row2)
                      print(row2[0])
                      password_list.append(str(row2[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(password_list)
    print(gmail_list.index(logu))
    print(password_list.index(passw))
    
    if gmail_list.index(logu)==password_list.index(passw):
        return render_template('index44.html')
    else:
        return jsonify({'result':'use proper  gmail and password'})
                  
                                               



                          
                     # print(value1[0:])
    
    
    
    

              
              # int_features3[0]==12345 and int_features3[1]==12345:
               #                      return render_template('index.html')
        
@app.route('/register',methods=['POST'])
def register():
    

    int_features2 = [str(x) for x in request.form.values()]
    #print(int_features2)
    #print(int_features2[0])
    #print(int_features2[1])
    r1=int_features2[0]
    print(r1)
    
    r2=int_features2[1]
    print(r2)
    logu1=int_features2[0]
    passw1=int_features2[1]
        
    

    

   # if int_features2[0]==12345 and int_features2[1]==12345:

    import MySQLdb


# Open database connection
    db = MySQLdb.connect("localhost","root",'',"ddbb" )

# prepare a cursor object using cursor() method
    cursor = db.cursor()
    cursor.execute("SELECT user FROM user_register")
    result1=cursor.fetchall()
              #print(result1)
              #print(gmail1)
    for row1 in result1:
                      print(row1)
                      print(row1[0])
                      gmail_list1.append(str(row1[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(gmail_list1)
    if logu1 in gmail_list1:
                      return jsonify({'result':'this gmail is already in use '})  
    else:

                  #return jsonify({'result':'this  gmail is not registered'})
              

# Prepare SQL query to INSERT a record into the database.
                  sql = "INSERT INTO user_register(user,password) VALUES (%s,%s)"
                  val = (r1, r2)
   
                  try:
   # Execute the SQL command
                                       cursor.execute(sql,val)
   # Commit your changes in the database
                                       db.commit()
                  except:
   # Rollback in case there is any error
                                       db.rollback()

# disconnect from server
                  db.close()
                 # return jsonify({'result':'succesfully registered'})
                  return render_template('login.html')

                      



@app.route('/production')
def production(): 
    return render_template('yield.html')



@app.route('/production1')
def production1(): 
    return render_template('recommendation.html')

@app.route('/production11',methods=['GET', 'POST'])
def production11(): 
    return render_template('disease.html')

@app.route('/production11/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Plant Disease Detection'

    if request.method == 'POST':
        #if 'file' not in request.files:
         #   return redirect(request.url)
            file = request.files.get('file')

            print(file)
        #if not file:
         #   return render_template('disease.html', title=title)
        #try:
            img1 = file.read()

            #print(img)

            prediction =pred_leaf_disease(img1)

            prediction = (str(disease_dic[prediction]))

            print(prediction)
            if prediction == "Apple___Apple_scab":
                precaution = "Precaution for Apple Scab: Remove and destroy diseased leaves as soon as they appear. Apply fungicide XYZ to control the disease. Ensure proper sanitation to prevent its spread."
            elif prediction == "Apple___Black_rot":
                precaution = "Precaution for Black Rot: Prune and remove infected branches and fallen leaves. Apply fungicide ABC following recommended guidelines. Keep the area clean and dry to reduce disease pressure."
            elif prediction == "Apple___Cedar_apple_rust":
                precaution = "Precaution for Cedar Apple Rust: Remove infected leaves and apply fungicide DEF as directed. Practice good orchard hygiene and avoid planting susceptible apple varieties nearby."
            elif prediction == "Corn___Cercospora_leaf_spot Gray_leaf_spot":
                precaution = "Precaution for Cercospora Leaf Spot Gray Leaf Spot in Corn: Remove and destroy infected corn leaves. Apply fungicide GHI as a preventive measure and practice crop rotation to reduce disease risk."
            elif prediction == "Corn___Common_rust":
                precaution = "Precaution for Common Rust in Corn: Remove and destroy infected corn leaves. Apply fungicide JKL as needed. Consider planting resistant corn varieties and practicing good field hygiene."
            elif prediction == "Corn___Northern_Leaf_Blight":
                precaution = "Precaution for Northern Leaf Blight in Corn: Remove and destroy infected corn leaves. Apply fungicide MNO for better control. Ensure proper spacing between corn plants to improve air circulation."
            elif prediction == "Grape___Black_rot":
                precaution = "Precaution for Black Rot in Grapes: Prune infected grapevines and remove fallen leaves. Apply fungicide PQR following recommended schedules. Maintain proper vineyard canopy management to reduce humidity."
            elif prediction == "Grape___Esca_(Black_Measles)":
                precaution = "Precaution for Esca (Black Measles) in Grapes: Prune and destroy infected vines. Apply fungicide STU for better control. Remove and burn pruned materials to reduce disease inoculum."
            elif prediction == "Strawberry___Leaf_scorch":
                precaution = "Precaution for Leaf Scorch in Strawberries: Remove infected strawberry leaves and apply fungicide VWX as needed. Ensure adequate spacing between plants for good air circulation."
            elif prediction == "Tomato___Bacterial_spot":
                precaution = "Precaution for Bacterial Spot in Tomatoes: Remove and destroy infected tomato leaves. Apply copper-based fungicide YZA. Ensure proper watering and avoid overhead irrigation."
            elif prediction == "Tomato___Early_blight":
                precaution = "Precaution for Early Blight in Tomatoes: Remove and destroy infected tomato leaves. Apply fungicide BCD as directed. Practice crop rotation and avoid planting tomatoes in the same area for consecutive seasons."
            elif prediction == "Tomato___Late_blight":
                precaution = "Precaution for Late Blight in Tomatoes: Remove and destroy infected tomato leaves. Apply fungicide EFG for better control. Ensure good air circulation in the tomato plants."
            elif prediction == "Tomato___Leaf_Mold":
                precaution = "Precaution for Leaf Mold in Tomatoes: Remove and destroy infected tomato leaves. Apply fungicide HIJ as needed. Maintain proper spacing between tomato plants to reduce humidity."
            elif prediction == "Tomato___Septoria_leaf_spot":
                precaution = "Precaution for Septoria Leaf Spot in Tomatoes: Remove and destroy infected tomato leaves. Apply fungicide KLM for control. Avoid overhead irrigation and ensure proper plant spacing."
            elif prediction == "Tomato___Spider_mites Two-spotted_spider_mite":
                precaution = "Precaution for Spider Mites (Two-Spotted Spider Mite) in Tomatoes: Use an insecticidal soap or oil to control mites. Remove heavily infested leaves. Ensure adequate humidity levels and avoid over-fertilization."
            elif prediction == "Tomato___Target_Spot":
                precaution = "Precaution for Target Spot in Tomatoes: Remove and destroy infected tomato leaves. Apply fungicide NOP as directed. Practice good sanitation and avoid planting tomatoes near other infected crops."
            elif prediction == "Tomato___Tomato_mosaic_virus":
                precaution = "Precaution for Tomato Mosaic Virus: Remove and destroy infected tomato plants to prevent the spread of the virus. Use virus-resistant tomato varieties if available."
            elif prediction == "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
                precaution = "Precaution for Tomato Yellow Leaf Curl Virus: Use virus-resistant tomato varieties. Remove and destroy infected tomato plants to prevent the virus's spread. Control whiteflies and maintain good weed management."
            else:
                precaution = "If you encounter any other plant diseases, please feel free to ask for advice and specific recommendations."

            return render_template('disease-result.html', prediction=prediction,precaution=precaution,title=title)
        #except:
         #   pass
    return render_template('disease.html', title=title)

@app.route('/production/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    print(features_list1,features_list)

    int_features = [str(x) for x in request.form.values()]



    print("output from web page  ",int_features)

    #features_list1[83]=int_features[0]



    state_ind=features_list.index(int_features[0])

    print(state_ind)

    features_list1[state_ind]=1



    print(int_features[1])


    


    #print(features_list.index('DHULE'))


    dist_ind=features_list.index(int_features[1])

    print(dist_ind)

    features_list1[dist_ind]=1

    season_ind=features_list.index(int_features[2])

    features_list1[season_ind]=1

    crop_ind=features_list.index(int_features[3])

    features_list1[crop_ind]=1

    crop_ind=features_list.index(int_features[5])

    features_list1[crop_ind]=1


    features_list1[0]=float(int_features[4])

   




    a=features_list1


    print(a)

    output_yield=yield_fn(features_list1)

    #features_list1=loaded_list

    random_float = round(random.uniform(0,0.09), 2)



    output_yield=random_float+output_yield




    return render_template('yield.html', prediction_text='Yield  will be    {} kg for Hectare '.format(output_yield[0]*1000))

@app.route('/crop')
def crop():
     return render_template('recommendation.html')



@app.route('/crop/predict1',methods=['POST'])
def predict1():
    '''
    For rendering results on HTML GUI
    '''
    int_features1 = [str(x) for x in request.form.values()]
   
    a1=int_features1[0]
    a2=int_features1[1]
    a3=int_features1[2]


    
    
    
   # int_features21 = np.array(int_features2)




  #  int_features11 = int_features21.reshape(1, -1)
   # prediction1 = model1.predict(int_features11)

    output1 = recondation_fn(a1,a2,a3)
   # resultcrop = {value:key for key, value in croplist.items()}
    print(output1)
    
    
 
    

    

    return render_template('recommendation.html', prediction1_text='You will get best Yield if you cultivate  {} '.format(output1[0]))



if __name__ == "__main__":
    app.run(debug=True)
