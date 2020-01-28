import requests
import sys, random, shutil
import datetime
from json.decoder import JSONDecodeError
import numpy as np

import tensorflow as tf
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from keras.layers import Dense, Input, Dropout, concatenate, PReLU

def get(year, day):
		try:
			url = 'https://www.openligadb.de/api/getmatchdata/bl1/'+str(year)+'/'+str(day)
			data = requests.get(url).json()
			return data
		except JSONDecodeError:
			return([])

def create():
	with open("data\\data.txt", 'a') as file:
		for year in range(2002,2020):
			print(">>>", year, "<<<")
			for day in range(1,35):
				data = get(year, day)
				for elem in data:
					if elem['MatchIsFinished']:
						for matchResult in elem['MatchResults']:
							if matchResult['ResultName'] == 'Endergebnis':
								print(day, '\t', elem['Team1']['TeamName'], '-', elem['Team2']['TeamName'], '\t', matchResult['PointsTeam1'], ':', matchResult['PointsTeam2'])
								file.write(str(year)+" "+str(day)+" "+str(elem['Team1']['TeamId'])+" "+str(elem['Team2']['TeamId'])+' '+str(matchResult['PointsTeam1'])+' '+str(matchResult['PointsTeam2'])+'\n')
			print()

def createModel(x):
	inputYear 	= Input(shape=(1,))
	inputDay 	= Input(shape=(1,))
	inputTeam1 	= Input(shape=(len(x[2]),))
	inputTeam2 	= Input(shape=(len(x[3]),))	
	inputPoints1 	= Input(shape=(1,))
	inputPoints2 	= Input(shape=(1,))	
	
	x = concatenate([inputYear, inputDay, inputTeam1, inputTeam2, inputPoints1, inputPoints2])		
	
	x = Dense(units=128, activation='relu')(x)
	x = Dropout(0.25)(x)
	x = Dense(units=64, activation='relu')(x)
	x = Dense(units=16, activation='relu')(x)
	x = Dense(units=8, activation='relu')(x)
	x1 = Dense(units=2, activation='linear', name='home')(x)
			
	model = Model(inputs=[inputYear, inputDay, inputTeam1, inputTeam2, inputPoints1, inputPoints2], outputs=[x1])
	model.compile(loss="mse", optimizer='adadelta', metrics=[])
	print(model.summary())
	return model

#create()

result = []

for seedVal in range(10):
	#seedVal = 1
	random.seed(seedVal)
	np.random.seed(seedVal)
	tf.set_random_seed(seedVal)

	with open("data\\data.txt") as file:
		lines = file.readlines()
		dct = {}
		
		for line in lines:
			sp = line.split(' ')
			year 	= int(sp[0])
			day 	= int(sp[1])
			team1 	= int(sp[2])
			team2 	= int(sp[3])
			score1 	= int(sp[4])
			score2 	= int(sp[5])
			
			if not year in dct:
				dct[year] = {}
				dct[year][0] = {}
				dct[year][0] = {}
			if not day in dct[year]:
				dct[year][day] = {}
			if score1 == score2:	
				dct[year][day][team1] = 1
				dct[year][day][team2] = 1
			elif score1 > score2:	
				dct[year][day][team1] = 3
				dct[year][day][team2] = 0
			else:
				dct[year][day][team1] = 0
				dct[year][day][team2] = 3
				
			if not team1 in dct[year][0]:
				dct[year][0][team1] = 0
			if not team2 in dct[year][0]:
				dct[year][0][team2] = 0	
	import copy
	dctSum = copy.deepcopy(dct)

	for year in dct.keys():
		for day in dct[year].keys():
			for team in dct[year][day].keys():
				for i in range(1,day):
					dctSum[year][day][team] += dct[year][i][team]

	with open("data\\data.txt") as file:
		lines = file.readlines()
		random.shuffle(lines)
		
		x1 = []
		x2 = []
		x3 = []
		x4 = []
		x5 = []
		x6 = []
		
		y = []
		y1 = []
		y2 = []
		teams = []
		weights = []
		for line in lines:
			sp = line.split(' ')
			team1 	= int(sp[2])
			team2 	= int(sp[3])
			
			if not team1 in teams:
				teams.append(team1)
			if not team2 in teams:
				teams.append(team1)
		
		for line in lines:
			sp = line.split(' ')
			year 	= int(sp[0])
			day 	= int(sp[1])
			team1 	= int(sp[2])
			team2 	= int(sp[3])
			score1 	= int(sp[4])
			score2 	= int(sp[5])
			
			ohYear = [0]*18
			ohYear[year-2002] = 1
			
			ohDay = [0]*34
			ohDay[day-1] = 1
			
			ohTeam1 = [0]*len(teams)
			ohTeam1[teams.index(team1)] = 1
			
			ohTeam2 = [0]*len(teams)
			ohTeam2[teams.index(team2)] = 1
			
			ohScore1 = [0]*10
			ohScore1[score1] = 1
			
			ohScore2 = [0]*10
			ohScore2[score2] = 1
			
			x1.append(np.array((year-2002)/18))
			x2.append(np.array(day/34))
			x3.append(np.array(ohTeam1))
			x4.append(np.array(ohTeam2))
			x5.append(np.array(dctSum[year][day-1][team1]))
			x6.append(np.array(dctSum[year][day-1][team2]))
			#y1.append(np.array([score1]))
			#y2.append(np.array([score2]))
			y.append([np.array(score1),np.array(score2)])
			
			weights.append(np.array(year-2001+day/34))
		
		id = "FB_"+datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
		tensorboard = TensorBoard(log_dir='./logs/'+id)
		checkpoint 	= ModelCheckpoint(filepath='./models/'+id, save_best_only=True, save_weights_only=True)
		shutil.copyfile(__file__, 'models/'+id+'.py') 
		#csv_logger = CSVLogger('C:/Users/Mr_X_/OneDrive/logs/log_'+id+'.log')
			
		model = createModel([x1[0],x2[0],x3[0],x4[0]])
		
		batch_size = 32
		epochs = 150
		model.fit(x=[x1,x2,x3,x4,x5,x6], y=np.array(y), batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[tensorboard, checkpoint], validation_split=0.2, sample_weight=np.array(weights), steps_per_epoch=None, validation_steps=None)#
		model.load_weights('models/'+id)
		year = 2019
		day = 20

		data = get(year, day)
		print()
		res = []
		for elem in data:
			if True:#not elem['MatchIsFinished']:
				print(elem['Team1']['TeamName'], 'vs', elem['Team2']['TeamName'])
		
				team1 = elem['Team1']['TeamId']
				team2 = elem['Team2']['TeamId']
		
				ohYear = [0]*18
				ohYear[year-2002] = 1
					
				ohDay = [0]*34
				ohDay[day-1] = 1
					
				ohTeam1 = [0]*len(teams)
				ohTeam1[teams.index(team1)] = 1
					
				ohTeam2 = [0]*len(teams)
				ohTeam2[teams.index(team2)] = 1
				
				x1 = []
				x2 = []
				x3 = []
				x4 = []
				for i in range(batch_size):
					x1.append(np.array((year-2002)/18))
					x2.append(np.array(day/34))
					x3.append(np.array(ohTeam1))
					x4.append(np.array(ohTeam2))
					x5.append(np.array(dctSum[year][day-1][team1]))
					x6.append(np.array(dctSum[year][day-1][team2]))
				pred = model.predict([x1,x2,x3,x4,x5,x6])[0]
				print(pred)
				res.append(pred)
		result.append(res)
np.save("test", result)
print("mean")
print(np.mean(result, axis=0))
print("std")
print(np.std(result, axis=0))