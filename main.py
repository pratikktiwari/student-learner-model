import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tkinter import *
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#open file for reading
data_url = ('Student Learner Model Survey.csv')
df = pd.read_csv(data_url)

#check whether data is loaded from csv or not

del df['Timestamp']
del df['Username']

print(df.head())


def fn_gender_vs_cgpa():
    plt.scatter(df['Current/ previous year CGPA'],df['Gender'],color='blue')
    plt.ylabel('Gender')
    plt.xlabel('CGPA')
    plt.show()

def fn_age_vs_cgpa():
    plt.scatter(df['Current/ previous year CGPA'],df['Age'],color='blue')
    plt.ylabel('Age')
    plt.xlabel('CGPA')
    plt.show()   

def fn_att_vs_cgpa():
    plt.scatter(df['Current/ previous year CGPA'],df['Current/ previous year attendance in percent'],color='blue')
    plt.ylabel('Attendence')
    plt.xlabel('CGPA')
    plt.show() 

def fn_sth_vs_cgpa():
    plt.scatter(df['Current/ previous year CGPA'],df['Approximated total hours of study per day'],color='blue')
    plt.ylabel('Hours of Study')
    plt.xlabel('CGPA')
    plt.show()

def fn_online_course_cgpa():
    plt.scatter(df['Current/ previous year CGPA'],df['Whether taking any online course through any of the platforms?'],color='blue')
    plt.ylabel('Online Course')
    plt.xlabel('CGPA')
    plt.show()

def fn_lecture_week_cgpa():
    plt.scatter(df['Current/ previous year CGPA'],df['Approximate number of lectures per week'],color='blue')
    plt.ylabel('Number of weekly lectures')
    plt.xlabel('CGPA')
    plt.show()

def fn_board_vs_cgpa():
    plt.scatter(df['Current/ previous year CGPA'],df['If attended high school, specify name of board'],color='blue')
    plt.ylabel('High School Board')
    plt.xlabel('CGPA')
    plt.show()

def fn_sports_vs_cgpa():
    plt.scatter(df['Current/ previous year CGPA'],df['Whether engaged in any sports activity?'],color='blue')
    plt.ylabel('Sports Engagement')
    plt.xlabel('CGPA')
    plt.show()

def fn_coaching_vs_cgpa():
    plt.scatter(df['Current/ previous year CGPA'],df['Have you attended coaching for any competitive exams earlier?'],color='blue')
    plt.ylabel('Coaching')
    plt.xlabel('CGPA')
    plt.show()

def predict_regr_att_vs_cgpa():
    msk = np.random.rand(len(df))<0.8
    cdf = df[['Current/ previous year CGPA','Current/ previous year attendance in percent']]
    x = np.array(cdf['Current/ previous year attendance in percent'])
    y = np.array(cdf['Current/ previous year CGPA'])
    
    train = cdf[msk]
    test = cdf[~msk]
    regr = LinearRegression()
    train_x = np.asanyarray(train[['Current/ previous year attendance in percent']])
    train_y  = np.asanyarray(train[['Current/ previous year CGPA']])

    test_x = np.asanyarray(test[['Current/ previous year attendance in percent']])
    test_y  = np.asanyarray(test[['Current/ previous year CGPA']])
    regr.fit(train_x, train_y)
    print('Coefficients: ', regr.coef_)
    print('Intercept   : ', regr.intercept_)

    cgpa_y_pred = regr.predict(test_x)
    mean_sq_err = mean_squared_error(test_y, cgpa_y_pred)
    r2_score_v = r2_score(test_y, cgpa_y_pred)
    print("Mean squared error: %.2f"% mean_sq_err)
    print('Variance score: %.2f' % r2_score_v)

    #change label text
    L3.config(text = 'Data Analytics')
    
    label_text = 'Equation for estimation: y = '+ str(regr.coef_) + ' + ' + str(regr.intercept_) + ' * x'
    L5.config(text = label_text)

    label_text = 'Mean Squared Error: '+ str(mean_sq_err)
    L6.config(text = label_text)

    label_text = 'Variance Score: '+ str(r2_score_v)
    L7.config(text = label_text)

    #plot training data set
    plt.subplot(2,1,1)
    plt.scatter(train_x, train_y,  color='black')
    y_hat = np.array(regr.coef_ + regr.intercept_*x)
    plt.plot(train_x,regr.coef_[0][0]*train_x+regr.intercept_[0], '-r')
    plt.xlabel('Current/ previous year attendance in percent') 
    plt.ylabel('Training- Current/ previous year CGPA')
    #plt.title('Training Plot')
    #plt.figure(figsize=(5, 5), dpi=80)

    #plot testing data set
    plt.subplot(2,1,2)
    plt.scatter(test_x, test_y,  color='black')
    plt.plot(test_x, cgpa_y_pred, color='blue', linewidth=3)
    plt.scatter(x, y, color = "m", marker = "o", s = 30)
    plt.xlabel('Current/ previous year attendance in percent') 
    plt.ylabel('Testing - Current/ previous year CGPA')
    #plt.title('Testing/ Prediction Plot')
    #plt.figure(figsize=(5, 5), dpi=80)
    plt.show()

def predict_based_on_user_input():
    cdf = df[['Current/ previous year CGPA','Current/ previous year attendance in percent']]
    regr = LinearRegression()
    data_x = np.asanyarray(cdf[['Current/ previous year attendance in percent']])
    data_y  = np.asanyarray(cdf[['Current/ previous year CGPA']])
    regr.fit(data_x, data_y)
    print('Coefficients: ', regr.coef_)
    print('Intercept   : ', regr.intercept_)

    #val_e1 = E1.get()
    val_e1 = e1_text_var.get()
    val_e1 = int(val_e1)
    
    predict_val = regr.intercept_[0] + regr.coef_[0][0]*val_e1
    label_text = 'Predicted value of CGPA is: '+str(predict_val)
    L9.config(text = label_text)
    
top = Tk()
#top.geometry('700x700')
photo = PhotoImage(file = 'image_ml1.PNG')
photo = photo.subsample(2)
lbl = Label(top,image = photo)
lbl.image = photo
lbl.grid(row=0,column=0, columnspan=12,rowspan=10)

L4 = Label(top, text = 'Student Learner Model Using Machine Learning')
L4.grid(row=11,column=0,columnspan=10,padx=15,pady=25)

B1 = Button(top, text = 'Gender vs CGPA',command=fn_gender_vs_cgpa)
B1.grid(row=12,column=0,padx=15,pady=15)

B2 = Button(top, text = 'Age vs CGPA',command=fn_age_vs_cgpa)
B2.grid(row=12,column=1,padx=15,pady=15)

B3 = Button(top, text = 'Attendence vs CGPA',command=fn_att_vs_cgpa)
B3.grid(row=12,column=2,padx=15,pady=15)

B4 = Button(top, text = 'Study Hours vs CGPA',command=fn_sth_vs_cgpa)
B4.grid(row=12,column=3,padx=15,pady=15)

B5 = Button(top, text = 'Online Course vs CGPA',command=fn_online_course_cgpa)
B5.grid(row=13,column=0,padx=15,pady=15)

B6 = Button(top, text = 'Lectures vs CGPA',command=fn_lecture_week_cgpa)
B6.grid(row=13,column=1,padx=15,pady=15)

B7 = Button(top, text = 'High School Board vs CGPA',command=fn_board_vs_cgpa)
B7.grid(row=13,column=2,padx=15,pady=15)

B8 = Button(top, text = 'Sports vs CGPA',command=fn_sports_vs_cgpa)
B8.grid(row=13,column=3,padx=15,pady=15)

B9 = Button(top, text = 'Coaching Ins vs CGPA',command=fn_coaching_vs_cgpa)
B9.grid(row=14,column=0,padx=15,pady=15)

B9 = Button(top, text = 'Predict Att vs CGPA',command=predict_regr_att_vs_cgpa)
B9.grid(row=14,column=1,padx=15,pady=15)

L3 = Label(top)
L3.grid(row=15,column=0,padx=15,pady=25)

L5 = Label(top)
L5.grid(row=16,column=0,columnspan=4,padx=5,pady=2)

L6 = Label(top)
L6.grid(row=17,column=0,columnspan=4,padx=5,pady=2)

L7 = Label(top)
L7.grid(row=18,column=0,columnspan=4,padx=5,pady=2)

L8 = Label(top, text = 'Enter value of Attendence to predict CGPA')
L8.grid(row=20,column=0,columnspan=4,padx=5,pady=10)


e1_text_var = StringVar()

E1 = Entry(top, bd=1, textvariable=e1_text_var)
E1.grid(row=21,column=0,padx=5,pady=5)
B10 = Button(top, text='Predict Now', command=predict_based_on_user_input)
B10.grid(row=21,column=1,padx=5,pady=5)

L9 = Label(top)
L9.grid(row=22,column=0,columnspan=3,padx=5,pady=2)

top.mainloop()


    

