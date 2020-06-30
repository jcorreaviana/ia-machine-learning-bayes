from sklearn.naive_bayes import MultinomialNB
import numpy as np
# characteristics - in order

# [oilBuyer, mouseBuyer, tireBuyer, gameBuyer, paintBuyer, fuelBuyer, headsetBuyer, energeticBuyer, bf3Player, carEntusiast]

carPerson1 = [1,0,1,0,1,1,0,1,0,1]
carPerson2 = [1,1,0,1,1,1,0,1,1,1]
carPerson3 = [0,0,1,0,1,1,1,1,0,1]
carPerson4 = [1,0,1,0,1,1,0,0,0,0]
carPerson5 = [0,0,0,1,1,1,0,1,1,1]
computerPerson1 = [0,1,0,1,0,0,1,1,1,0]
computerPerson2 = [0,1,0,1,0,0,1,1,1,1]
computerPerson3 = [0,1,1,1,0,1,1,0,0,0]
computerPerson4 = [1,0,0,1,1,0,0,1,1,1]
computerPerson5 = [0,0,0,1,1,1,0,1,1,1]

data = [carPerson1, carPerson2, carPerson3, carPerson4, carPerson5, computerPerson1, computerPerson2, computerPerson3, computerPerson4, computerPerson5]
dataClassification = [1,1,1,1,1,0,0,0,0,0]

testPersonCar1 = [1,0,1,0,1,1,0,1,1,1]
testPersonCar2 = [1,0,1,1,1,1,1,1,1,1]
testPersonCar3 = [0,0,1,1,1,1,0,1,0,1]
testPersonComputer1 = [0,1,0,1,1,0,1,1,1,0]
testPersonComputer2 = [1,1,0,1,1,1,1,1,1,1]
testPersonComputer3 = [0,0,0,1,0,0,0,0,1,0]

dataTests = [testPersonCar1, testPersonCar2, testPersonCar3, testPersonComputer1, testPersonComputer2, testPersonComputer3];
testClassification = [1,1,1,0,0,0]

model = MultinomialNB()

model.fit(data, dataClassification)
result = model.predict(dataTests)

difference = result - testClassification

success = [d for d in difference if d == 0]
score = len(success)
elements = len(dataTests)

accuracy = 100 * (score/elements)
error = 100 - accuracy
print("acurracy rate: {0}".format(str(accuracy)))
print("error rate: {0}".format(str(error)))