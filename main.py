import math
from statistics import stdev
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
            min1 = min(data)
            max1 = max(data)
            if max1 - min1 == 0:
                continue  # Skip normalization if the range is zero
            x[column] = (data - min1) / (max1 - min1)

    @staticmethod
    def split(x, percentage):
        num_rows = len(x)
        split_index = int(num_rows * percentage)
        training = x.iloc[:split_index]
        testing = x.iloc[split_index:]
        return training, testing

    # Metoda do czyszczenia rekordow, które mają zbyt odstające wartości,
    # iterujemy po wszystkich kolumnach i czyscimy metodą 'IQR'
    # czyli Interquartile Range (IQR) method, pol. (rozstęp ćwiartkowy),
    # czyli bierzemy różnice między trzecim a pierwszym kwartylem czyli obszar,
    # w którym mieści się 50% obserwacji
    def cleanColumn(data, columns, thr=2):
        column_desc = data[columns].describe()

        q3 = column_desc[6]
        q1 = column_desc[4]
        IQR = q3 - q1

        top_limit_clm = q3 + thr * IQR
        bottom_limit_clm = q1 - thr * IQR

        filter_clm_bottom = bottom_limit_clm < data[columns]
        filter_clm_top = data[columns] < top_limit_clm

        filters = filter_clm_bottom & filter_clm_top

        data = data[filters]

        print("{} of dataset after column {}".format(data.shape, columns))

        return data

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

# 1. OPIS OGÓLNY DANYCH W BAZIE

print()
print()
print("Ogólne informacje dot. :")
print()
print()

diabetes.info()

print()
print()
print("Opis bazy: (średnia,  odchylenie standardowe, min, max)")
print()
print()
print(diabetes.describe())

print()
print()
print("Korelacja danych w bazie: ")
print()
print()

print(diabetes.corr())




# wizualizacja graficzna korelacji danych
f, ax = plt.subplots(figsize = (10,10))
sns.heatmap(diabetes.corr(), annot = True, linewidths = 0.5, linecolor = "black", fmt = ".4f", ax = ax)


#wizualuzacja graficzna danych w sposób punkowy
# sns.pairplot(diabetes, hue = "Outcome", kind='reg')
# plt.show()

# rozmiar danych (rekordy x kolumny)

print()
print()

print("Rozmiar bazy danych rekordy x kolumny")

print()
print()

print(diabetes.shape)


# 2. ANALIZA DANYCH W BAZIE

# Graficzne zestawienie wyników diagnoz dla całej bazy:

plt.figure()
sns.countplot(data=diabetes, x='Outcome')

# Ponieważ zauważyliśmy, że ilość ciąż ma znikomy wpływ na wynik,
# sprawdzamy ile mamy kobiet z iloma ciazami

print()
print()
print("Ilość ciąż pogrupowana według ilości kobiet: ")
print()
print()
print(diabetes["Pregnancies"].value_counts())


plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
sns.countplot(data=diabetes, x='Pregnancies')
plt.xlabel('Number of Pregnancies')
plt.ylabel('Number of Women')
plt.title('Number of Women grouped by number of pregnancies')


print()
print()
print("Zauważyliśmy sporo zer w kolumnie 'Insulin', sprawdźmy to: ")
print()
print()

zero_Insulin = 0
zero_Outcome_Insulin = 0
one_Outcome_Insulin = 0

for i in range(len(diabetes["Insulin"])):
    if (diabetes["Insulin"][i] == 0):
        zero_Insulin += 1
        if (diabetes["Outcome"][i] == 0):
            zero_Outcome_Insulin += 1

        if (diabetes["Outcome"][i] == 1):
            one_Outcome_Insulin += 1

print("Ilość 0 w kolumnie 'Insulin' dla wszystkich wyników ", zero_Insulin)
print("Ilość 0 w kolumnie 'Insulin' dla braku cukrzycy: ", zero_Outcome_Insulin)
print("Ilość 0 w kolumnie 'Insulin' dla zdiagnozowanej cukrzycy ", one_Outcome_Insulin)

# 3. BADAMY WARTOŚCI KRAŃCOWE, I CZYŚCIMY Z NICH BAZE

#tutaj mamy wykresy ,które nam pokazują odstające wartości
for c in diabetes.columns:
    plt.figure()
    sns.boxplot(x = c, data = diabetes, orient = "v")


# UWAGA! Zauważamy, że Insulina i DiabetesPedigreeFunction ma dużo
# odstających wartości, więc czyścimy kolumny z odstających wartości
# implemeentujemy metode cleanColumn(data,columns), która czysci
# wszystkie kolumny metodą IQR (Interquartile Range  method),
# pol (metoda rozstępu ćwiartkowego)

print()
print()
print("Metoda IQR do czyszczenia danych: PRZEBIEG KOLUMNA PO KOLUMNIE:")
print()
print()


for i in diabetes.columns:
    diabetes = DataProcessing.cleanColumn(diabetes,i)

print()
print()
print("Nowy wymiar po czyszczeniu kolumn metodą IQR: ", diabetes.shape)
print()
print()


DataProcessing.shuffle(diabetes)

print("Nieznormalizowane")
train, test = DataProcessing.split(diabetes, 0.7)

tmp = NaiveBayes.classify(train, test.iloc[0])
print(tmp)

counter = 0
for i in range(len(test)):
    tmp = NaiveBayes.classify(train, test.iloc[i])[0]
    if tmp == test.iloc[i]['Outcome']:
        counter += 1
dokladnosc = float(counter)/len(test) * 100
print(dokladnosc, "%")

print("Znormalizowane")

DataProcessing.normalization(diabetes)
tmp = NaiveBayes.classify(train, test.iloc[0])
print(tmp)

counter = 0
for i in range(len(test)):
    tmp = NaiveBayes.classify(train, test.iloc[i])[0]
    if tmp == test.iloc[i]['Outcome']:
        counter += 1
dokladnosc = float(counter)/len(test) * 100
print(dokladnosc, "%")

plt.show()