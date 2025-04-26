from tkinter import messagebox
from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

main = Tk()
main.title("AI-Powered Vehicle Model Forecasting through Advanced Pattern Recognition Technique")
main.geometry("1300x1200")

global filename
global X, Y
global model
global X_train, X_test, y_train, y_test
accuracy = []
global XX
global classifier 

names = ['AM General Hummer SUV 2000', 'Acura RL Sedan 2012', 'Acura TL Sedan 2012', 'Acura TL Type-S 2008', 'Acura TSX Sedan 2012']

def uploadDataset():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END, 'Dataset loaded\n')
    X = np.load("model/X.txt.npy")
    Y = np.load("model/Y.txt.npy")
    X = np.asarray(X)
    Y = np.asarray(Y)
    img = X[20].reshape(64, 64, 3)
    cv2.imshow('ff', cv2.resize(img, (250, 250)))
    cv2.waitKey(0)

def SVMCNN():
    global classifier 
    X = np.load("model/X.txt.npy")
    Y = np.load("model/Y.txt.npy")
    print(X.shape)
    print(Y.shape)

    # Reshape and apply PCA
    temp = X
    XX = np.reshape(temp, (temp.shape[0], (temp.shape[1] * temp.shape[2] * temp.shape[3])))
    pca = PCA(n_components=180)
    XX = pca.fit_transform(XX)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2)

    # SVM classifier
    cls = svm.SVC()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test, predict) * 100
    accuracy.append(acc)
    text.insert(END, 'SVM Prediction Accuracy: ' + str(acc) + "\n")

    # CNN classifier setup
    Y1 = to_categorical(Y)  # One-hot encoding of labels
    cnn = Sequential()
    cnn.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Convolution2D(32, 3, 3, activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(units=256, activation='relu'))  # Corrected layer argument
    cnn.add(Dense(units=5, activation='softmax'))  # Adjust the number of classes accordingly
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Fit the model
    cnn_history = cnn.fit(X, Y1, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
    cnn_history = cnn_history.history
    acc = cnn_history['accuracy'][-1] * 100  # Get accuracy from the last epoch
    accuracy.append(acc)
    text.insert(END, 'CNN Prediction Accuracy: ' + str(acc) + "\n\n")

    classifier = cnn  # Ensure classifier is assigned here

    # Plot the accuracy comparison
    bars = ('SVM', 'CNN')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, [accuracy[-2], accuracy[-1]])
    plt.xticks(y_pos, bars)
    plt.show()
    plt.title('SVM & CNN Accuracy Performance Graph')
    plt.show()

def predict():
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (64, 64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 64, 64, 3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img / 255
    preds = classifier.predict(img)
    predict = np.argmax(preds)

    img = cv2.imread(filename)
    img = cv2.resize(img, (800, 400))
    cv2.putText(img, 'Car Model Predicted as : ' + names[predict], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow('Car Model Predicted as : ' + names[predict], img)
    cv2.waitKey(0)

def graph():
    bars = ('SVM', 'CNN')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, accuracy)
    plt.xticks(y_pos, bars)
    plt.show()
    plt.title('SVM & CNN Accuracy Performance Graph')
    plt.show()

def close():
    main.destroy()

font = ('times', 15, 'bold')
title = Label(main, text='AI-Powered Vehicle Model Forecasting through Advanced Pattern Recognition Technique')
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0, y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Cars Dataset", command=uploadDataset)
uploadButton.place(x=20, y=100)
uploadButton.config(font=ff)

cnnButton = Button(main, text="Run SVM & CNN Algorithms", command=SVMCNN)
cnnButton.place(x=20, y=200)
cnnButton.config(font=ff)

predictButton = Button(main, text="Prediction Model", command=predict)
predictButton.place(x=20, y=350)
predictButton.config(font=ff)

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=20, y=400)
graphButton.config(font=ff)

font1 = ('times', 12, 'bold')
text = Text(main, height=30, width=85)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450, y=100)
text.config(font=font1)

main.config()
main.mainloop()
