from sklearn.naive_bayes import GaussianNB
'''
https://pythonprogramminglanguage.com/naive-bayes-classifier/
'''
# create naive bayes classifier
gaussian_nb = GaussianNB()

# The training set (X) simply consists of length, weight and shoe size.
# Y contains the associated labels (male or female).
# create dataset
X = [[121, 80, 44],
     [180, 70, 43],
     [166, 60, 38],
     [153, 54, 37],
     [166, 65, 40],
     [190, 90, 47],
     [175, 64, 39],
     [174, 71, 40],
     [159, 52, 37],
     [171, 76, 42],
     [183, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

# train classifier with dataset
gaussian_nb = gaussian_nb.fit(X, Y)

# predict using classifier
prediction = gaussian_nb.predict([[190, 70, 43]])
print(prediction)
