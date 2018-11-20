import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import KFold
from tkinter import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


FP = 0
FN = 0
TN = 0
TP = 0

data = pd.read_csv('heart.csv', header=None)

dj = pd.DataFrame(data) #data frame

yj = dj.iloc[:, 13]
yj = yj-1

def chol_age():
	x = dj.iloc[:, 0:5]
	x = x.drop(x.columns[1:4], axis=1)
	chol_avgs = x.groupby(0, sort=True).mean()
	ages = (chol_avgs[4].index.values)
	avgs = (chol_avgs[4].values)
	plt.plot(ages,avgs,'g-')
	plt.title('Variation of Cholestrol Levels with Age')
	plt.xlabel('Age(years)')
	plt.ylabel('Serum Cholestrol in mg/dl')
	plt.show()

def heart_atrack_heart_rate_bp():
	x = dj.iloc[:, 0:14]
	x[14] = np.round(dj[3], -1)

	x_dis = x[x[13] == 2]
	bp_set_dis = x_dis.groupby(14, sort=True)
	nums_dis = (bp_set_dis.count()[0]).index.values
	bps_dis = (bp_set_dis.count()[0]).values
	bar2 = plt.bar(nums_dis+2, bps_dis, color='r', width=2)

	x_nor = x[x[13] == 1]
	bp_set_nor = x_nor.groupby(14, sort=True)
	nums_nor = (bp_set_nor.count()[0]).index.values
	bps_nor = (bp_set_nor.count()[0]).values
	bar1 = plt.bar(nums_nor, bps_nor, color='g', width=2)

	plt.title('Resting blood pressure as heart risk indicator')
	plt.xlabel('Resting Blood Pressure Bucket')
	plt.ylabel('Number of Patients')

	plt.legend((bar1[0], bar2[0]), ('Safe', 'At Risk'))
	plt.show()

def pie_chart_chest_pain():
	x = dj.iloc[:, 0:3]
	sets = x.groupby(2).count()
	fin_lab = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptotic']
	values = (sets[0].values)
	plt.pie(values, labels=fin_lab, colors=['yellowgreen', 'gold', 'lightskyblue', 'lightcoral'], explode = [0,0.2,0,0], shadow=True, autopct='%1.1f%%', startangle=90)
	plt.title('Chest Pain Types')
	plt.show()

def scatter_chart():
	x = dj.iloc[:, 0:13]
	sc = plt.scatter(x[7],x[4], c=yj, cmap='summer')
	plt.title('Dataset Scatter')
	classes = ['Safe', 'At Risk']
	class_colours = ['g','y']
	recs = []
	for i in range(0,len(class_colours)):
		recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
	plt.legend(recs, classes)
	plt.show()
    
def graphx():
    scatter_chart()
    heart_atrack_heart_rate_bp()
    pie_chart_chest_pain()
    chol_age()
    
def nbkmh(train_index, test_index):
	#extracting columns x and y separately for kmeans and naive bayes classifiers
	x_kmeans = df.iloc[:, 0:5]
	x_kmeans = x_kmeans.drop(x_kmeans.columns[1:3], axis=1)
	#x_kmeans = pd.DataFrame(scale(x_kmeans))
	#print(x_kmeans)

	x_naive = df.iloc[:, 0:13]

	y = df.iloc[:, 13]
	y = y-1

	y_train = pd.Series(y.iloc[train_index])
	y_test = pd.Series(y.iloc[test_index])
	#print(pd.Series(y_train.iloc[6]))

	x_train_kmeans = x_kmeans.iloc[train_index, :]
	x_test_kmeans = x_kmeans.iloc[test_index, :]

	x_train_naive = x_naive.iloc[train_index, :]
	x_test_naive = x_naive.iloc[test_index, :]


	#Kmeans model for the processed data
	clusters = 5
	global model_kmeans
	model_kmeans = KMeans(init='k-means++', n_clusters=clusters, n_init=10,random_state=10000)
	#print(model_kmeans)
	model_kmeans.fit(x_train_kmeans)
	#print(model_kmeans)
	kmean_predictions = model_kmeans.predict(x_train_kmeans)
	#print(kmean_predictions)

	#building datset according to clusters
	x = [pd.DataFrame() for ii in range(0,clusters)]
	#print(x)
	y = [pd.Series() for ii in range(0,clusters)]
	#print(y)

	for kmean_prediction,i in zip(kmean_predictions, range(len(x_train_kmeans))):
		row_x =  x_train_naive.iloc[i, :]
		row_y = pd.Series(y_train.iloc[i])
		index = int(kmean_prediction)
		x[index] = x[index].append(row_x, ignore_index=True)
		y[index] = y[index].append(row_y)

	#applying naive bayes classifier
	global clstr_n
	clstr_n = [MultinomialNB(alpha=2,fit_prior=True) for ii in range(0,clusters)]

	for i in range(0,clusters):
		clstr_n[i].fit(x[i], y[i])

	#calculating predictions for the testing based on the hybrid algorithm
	predicts = []
	c=0
	for i in range(len(x_test_kmeans)):
		prediction = model_kmeans.predict(x_test_kmeans.iloc[i, :].reshape(1,-1))
		#print(prediction)
		prediction = int(prediction)
		#print(prediction)
		pred_naive = clstr_n[prediction].predict(x_test_naive.iloc[i, :].reshape(1,-1))
		#print(prediction)
		predicts.append(pred_naive)
		if pred_naive == y_test.iloc[i]:
			c+=1

	print ((c*100.0)/len(x_test_kmeans))
	# print ("Test set accuracy : ",  ((c*100.0)/len(x_test_kmeans)))
	


	#metrics
	predicts = np.array(predicts)
	cm = metrics.confusion_matrix(y_test, predicts)/len(y_test)
	#print (cm)
	global FP
	global FN
	global TN
	global TP

	FP += cm[0][0]
	FN += cm[1][0]
	TN += cm[0][1]
	TP += cm[1][1]

	return ((c*100.0)/len(x_test_kmeans))

def userin():
    age = e2.get()
    gen = e3.get()
    cp = e4.get()
    bps = e5.get()
    chloe = e6.get()
    fbs = e7.get()
    ecg = e8.get()
    thalach = e9.get()
    ange = e10.get()
    opeak = e11.get()
    slope = e12.get()
    ca = e13.get()
    thal = e14.get()
    
    predictionx = model_kmeans.predict(pd.Series([age,bps,chloe]).reshape(1,-1))
    #print(predictionx)
    predictionx = int(predictionx)
    #print(predictionx)
    pred_naivex = clstr_n[predictionx].predict(pd.Series([age,gen,cp,bps,chloe,fbs,ecg,thalach,ange,opeak,slope,ca,thal]).reshape(1,-1))
    if(pred_naivex==1):
        print("Person has heart disease",pred_naivex)
    else:
        print("Person doesnot have heart disease",pred_naivex)

    
def main():
	scores = []
	#importing dataset and converting to datasframe
	file = e1.get()
	global df,data
	data = pd.read_csv(file, header=None)
	df = pd.DataFrame(data)
	#data frame

    
	kf = KFold(n=df.shape[0], n_folds=10)

	for (train_index,test_index),i in zip(kf,range(0,10)):
		print("Iteration " + str(i+1) + " : ")
		scores.append(nbkmh(train_index, test_index))
	print("\n 10 Fold Accuracy",np.array(scores).mean())
	print("FP", FP*10)
	print("FN", FN*10)
	print("TN", TN*10)
	print("TP", TP*10)
	#userin()

window=Tk()

l1=Label(window,text="Enter CSV File")
l1.grid(row=1,column=0)

title_text=StringVar()
e1=Entry(window,textvariable=title_text)
e1.grid(row=1,column=1)

b1=Button(window,text="Submit", width=12,command=main)
b1.grid(row=2,column=1,columnspan=2)

l2=Label(window,text="Age")
l2.grid(row=3,column=0)

title_text=StringVar()
e2=Entry(window,textvariable=title_text)
e2.grid(row=3,column=1)

l3=Label(window,text="Gender")
l3.grid(row=4,column=0)

title_text=StringVar()
e3=Entry(window,textvariable=title_text)
e3.grid(row=4,column=1)

l4=Label(window,text="Cp")
l4.grid(row=5,column=0)

title_text=StringVar()
e4=Entry(window,textvariable=title_text)
e4.grid(row=5,column=1)

l5=Label(window,text="Bps")
l5.grid(row=6,column=0)

title_text=StringVar()
e5=Entry(window,textvariable=title_text)
e5.grid(row=6,column=1)

l6=Label(window,text="Chloe")
l6.grid(row=7,column=0)

title_text=StringVar()
e6=Entry(window,textvariable=title_text)
e6.grid(row=7,column=1)

l7=Label(window,text="FBS")
l7.grid(row=8,column=0)

title_text=StringVar()
e7=Entry(window,textvariable=title_text)
e7.grid(row=8,column=1)

l8=Label(window,text="ECG")
l8.grid(row=9,column=0)

title_text=StringVar()
e8=Entry(window,textvariable=title_text)
e8.grid(row=9,column=1)

l9=Label(window,text="Thalach")
l9.grid(row=10,column=0)

title_text=StringVar()
e9=Entry(window,textvariable=title_text)
e9.grid(row=10,column=1)

l10=Label(window,text="EX an")
l10.grid(row=11,column=0)

title_text=StringVar()
e10=Entry(window,textvariable=title_text)
e10.grid(row=11,column=1)

l11=Label(window,text="Old peak")
l11.grid(row=12,column=0)

title_text=StringVar()
e11=Entry(window,textvariable=title_text)
e11.grid(row=12,column=1)

l12=Label(window,text="Slope")
l12.grid(row=13,column=0)

title_text=StringVar()
e12=Entry(window,textvariable=title_text)
e12.grid(row=13,column=1)

l13=Label(window,text="Ca")
l13.grid(row=14,column=0)

title_text=StringVar()
e13=Entry(window,textvariable=title_text)
e13.grid(row=14,column=1)

l14=Label(window,text="Thal")
l14.grid(row=15,column=0)

title_text=StringVar()
e14=Entry(window,textvariable=title_text)
e14.grid(row=15,column=1)

b2=Button(window,text="Predict", width=12,command=userin)
b2.grid(row=16,column=1,columnspan=2)

b3=Button(window,text="Graph", width=12,command=graphx)
b3.grid(row=17,column=1,columnspan=2)

window.mainloop()
