import math
from statistics import stdev

import numpy
import numpy as np    # do odczytu z pliku
import pandas as pd   # do analiz
import random as rn   # do randomowych liczb
import seaborn as sns # do generowania wykresow szybkich + barwy ma (lepsze niz matplotlib)
import matplotlib.pyplot as plt
from numpy import mean
import warnings
warnings.filterwarnings("ignore")


class DataProcessing:
    @staticmethod
    def shuffle(x):
        for i in range(len(x)-1, 0, -1):
            j = rn.randint(0, i-1)
            x.iloc[i], x.iloc[j] = x.iloc[j], x.iloc[i]

    @staticmethod
    def normalization(x):
        # object to obiekt czyli string bo string w pythonie to object
        values = x.select_dtypes(
            exclude="object")  # w skrocie to wyrzuca z bazy nienumeryczne dane np string, bo by sie nie dalo zrobic obliczen
        columnNames = values.columns.tolist()
        for column in columnNames:
            data = x.loc[:,column]  # to prawdopodobnie bierze wszystkie wartosci z danej kolumny i pozniej bierze min i max wartosc z niej
            min1 = min(data)
            max1 = max(data)
            for row in range(len(x)):
                xprim = (x.at[row, column] - min1) / (max1 - min1)
                x.at[row, column] = xprim  # tu podmiana wartsci oryginalnej na przeskalowaną

    @staticmethod
    def split(x, percentage):
        num_rows = len(x)
        split_index = int(num_rows * percentage)
        training = x.iloc[:split_index]
        testing = x.iloc[split_index:]
        return training, testing


class NaiveBayes:
    @staticmethod
    def classify(x, sample):
        probability = []
        #dla kazdej klasy
        classNames = x['variety'].unique().tolist()
        for className in classNames:
            columnNames = x.columns.tolist()[:4] # wszystkie elementy do tego
            prob = 1
            tmp = x[x["variety"] == className]  # tylko elementy klasy className
            for columnName in columnNames:
                data = tmp.loc[:, columnName] #wyciągamy wszystkie wartosci z columnName
                mu = mean(data)
                sigma = stdev(data)
                #prawdopodobienstwo teraz
                prob *= 1/(sigma * math.sqrt(np.pi*2)) * np.exp(-0.5*((sample[columnName]-mu)/sigma)**2)
            prob *= len(tmp)/len(x) # prob = prob * (ilosc elementow danej klasy / ilosc wszystkich (poczatkowe prawdopod.))
            probability.append([className, prob])
    #znajdz maksymalne prawdopodobienstwo w tablicy probability i zwroc nazwe klasy
        maxprobNameAndValue = max(probability, key = lambda x:x[1])
        return maxprobNameAndValue


# wczytanie bazy

diabetes = pd.read_csv("diabetes.csv")

# diabetes.info()
print(diabetes.head())

# zastanow sie jak skategoryzowac kolumny od pregnacies do skinThickness i napisz funkcje kategoryzujące je
# ( prawdopodobnie kazda kolumna to inna funkcja)