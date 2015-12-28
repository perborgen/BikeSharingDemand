import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import datetime as DT

quarter_of_year = 7889031
epoch_start_time = DT.date(2011,01,20).strftime('%s')

data = pd.read_csv('train.csv', header=0)
X = data.drop(["datetime","casual","registered","count"], axis=1)
Y = data["count"]

# create two new fields, weekday (0-6) and isWeekend (0,1)
def datetimeToWeekday(datetime):
	date = datetime.split('-')
	date = DT.date(int(date[0]),int(date[1]),int(date[2]))
	weekday = date.weekday()
	isWeekend = 0
	if weekday == 5 or weekday == 6:
		isWeekend = 1
	return [weekday, isWeekend]

# create a feature that contains quarters since start, as the demand service grows in popularity
def datetimeToQuarterSince(datetime):
	date = datetime.split('-')
	epoch_now = DT.date(int(date[0]),int(date[1]),int(date[2])).strftime('%s')
	difference = int(epoch_now) - int(epoch_start_time)
	quarters_since = int(difference / quarter_of_year)
	return quarters_since

# create a feature which contains the hour of the day
def datetimeToHourArray(datetime):
	datetimeArray = []
	datetimeArray.append(int(datetime))
	return datetimeArray

# non used feature: month
def datetimeToMonthArray(datetime):
	datetimeMonthArray = []
	datetimeMonthArray.append(int(datetime))
	return datetimeMonthArray


# reformating the data
def reformat(data, isTrain):
	if isTrain == True:
		X = data.drop(["datetime","casual","registered","count"], axis=1)
	elif isTrain == False:
		X = test_data.drop(["datetime"], axis=1)
	X = np.array(X)
	datetime = data["datetime"]
	HourArrays = []
	MonthArrays = []
	daysSinceStartArray = []
	daysSinceStart = 0
	weekdays = []
	quarters = []
	for index in xrange(len(X)):
		HourArray = datetimeToHourArray(datetime[index][11:13])
		HourArrays.append(HourArray)
		daysSinceStartArray.append([daysSinceStart])
		daysSinceStart += 1
		weekday = datetimeToWeekday(datetime[index][0:10])
		quarters_since = datetimeToQuarterSince(datetime[index][0:10])
		weekdays.append(weekday)
		quarters.append([quarters_since])
	HourArrays = np.array(HourArrays)
	X = np.concatenate((X, HourArrays, weekdays, quarters), axis=1)
	return X
 
X = reformat(data, True)

print X[0]
# Tried out various models, but Random Forest worked best
#clf = linear_model.Lasso(alpha = 0.1)
#clf = linear_model.Ridge(alpha = .5)
#clf = KNeighborsRegressor(n_neighbors=4)
clf = RandomForestRegressor(n_estimators=1000, min_samples_leaf=4)
clf.fit(X, Y)


# Importing the test data and writing the results to the submissions file
test_data = pd.read_csv('test.csv', header=0)
test_data_f = reformat(test_data, False)
predictions = clf.predict(test_data_f)

# adding the space in " datetime" to avoid "count" to be the first column in submission.csv
output = pd.DataFrame(data={"count": predictions, " datetime": test_data["datetime"]})
output.to_csv("submission.csv", index=False, quoting=3 )
