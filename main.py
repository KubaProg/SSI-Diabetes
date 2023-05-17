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
        values = x.select_dtypes(exclude="object")
        columnNames = values.columns.tolist()
        for column in columnNames:
            data = x[column]
            non_zero_data = data[data != 0]
            min1 = min(non_zero_data)
            max1 = max(non_zero_data)
            if max1 - min1 == 0:
                continue  # Skip normalization if the range is zero
            for row, value in data.iteritems():
                if value != 0:
                    xprim = (value - min1) / (max1 - min1)
                    x.at[row, column] = xprim

    @staticmethod
    def split(x, percentage):
        num_rows = len(x)
        split_index = int(num_rows * percentage)
        training = x.iloc[:split_index]
        testing = x.iloc[split_index:]
        return training, testing
    @staticmethod
    def get_records_with_max_6_pregnancies(diabetes):
        max_pregnancies = 6
        filtered_records = diabetes[diabetes['Pregnancies'] <= max_pregnancies]

        return filtered_records

class NaiveBayes:
    @staticmethod
    def classify(x, sample):
        probability = []
        classNames = x['Outcome'].unique().tolist()
        for className in classNames:
            columnNames = x.columns.tolist()[:8]
            prob = 1
            tmp = x[x["Outcome"] == className]
            for columnName in columnNames:
                data = tmp.loc[:, columnName]
                mu = mean(data)
                sigma = stdev(data)

                if columnName == "Glucose":
                    prob *= 2 / (sigma * math.sqrt(np.pi * 2)) * np.exp(-0.5 * ((sample[columnName] - mu) / sigma) ** 2)
                elif columnName == "BMI":
                    prob *= 1.5 / (sigma * math.sqrt(np.pi * 2)) * np.exp(-0.5 * ((sample[columnName] - mu) / sigma) ** 2)
                elif columnName == "Pregnancies":
                    prob *= 0.8 / (sigma * math.sqrt(np.pi * 2)) * np.exp(-0.5 * ((sample[columnName] - mu) / sigma) ** 2)
                elif columnName == "DiabetesPedigreeFunction":
                    prob *= 1.5 / (sigma * math.sqrt(np.pi * 2)) * np.exp(-0.5 * ((sample[columnName] - mu) / sigma) ** 2)
                else:
                    prob *= 1 / (sigma * math.sqrt(np.pi * 2)) * np.exp(-0.5 * ((sample[columnName] - mu) / sigma) ** 2)

            prob *= len(tmp) / len(x)
            probability.append([className, prob])

        maxprobNameAndValue = max(probability, key=lambda x: x[1])
        return maxprobNameAndValue

# wczytanie bazy

diabetes = pd.read_csv("diabetes.csv")

print()
print("Ogólne informacje dot. :")
print()

diabetes.info()

print()
print("Opis bazy: (średnia,  odchylenie standardowe, min, max)")
print()

print(diabetes.describe())

print()
print("Korelacja danych w bazie: ")
print()

print(diabetes.corr())


# wizualizacja graficzna korelacji danych
f, ax = plt.subplots(figsize = (10,10))
sns.heatmap(diabetes.corr(), annot = True, linewidths = 0.5, linecolor = "black", fmt = ".4f", ax = ax)
plt.show()

#wizualuzacja graficzna danych w sposób punkowy
# sns.pairplot(diabetes, hue = "Outcome", kind='reg')
# plt.show()

# rozmiar danych (rekordy x kolumny)
print("Rozmiar bazy danych rekordy x kolumny")
print(diabetes.shape)




# diabetes = DataProcessing.get_records_with_max_6_pregnancies(diabetes)

# DataProcessing.shuffle(diabetes)
#
# print("Nieznormalizowane")
# train, test = DataProcessing.split(diabetes, 0.7)
#
# tmp = NaiveBayes.classify(train, test.iloc[0])
# print(tmp)
#
# counter = 0
# for i in range(len(test)):
#     tmp = NaiveBayes.classify(train, test.iloc[i])[0]
#     if tmp == test.iloc[i]['Outcome']:
#         counter += 1
# dokladnosc = float(counter)/len(test) * 100
# print(dokladnosc, "%")
#
# print("Znormalizowane")
#
# DataProcessing.normalization(diabetes)
# tmp = NaiveBayes.classify(train, test.iloc[0])
# print(tmp)
#
# counter = 0
# for i in range(len(test)):
#     tmp = NaiveBayes.classify(train, test.iloc[i])[0]
#     if tmp == test.iloc[i]['Outcome']:
#         counter += 1
# dokladnosc = float(counter)/len(test) * 100
# print(dokladnosc, "%")
#
