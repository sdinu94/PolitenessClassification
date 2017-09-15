import re
import nltk as nk
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r'C:\Users\sdinu\OneDrive\Documents\02 MS BA\03 Spring 2017\IDS 566 Adv Text Analytics\Project\train.csv')
Politeness=df["politeness"]

uniques=[]
for word in Politeness:
    if word not in uniques:
        uniques.append(word)

counts=[]
for unique in uniques:
    count=0
    for letter in Politeness:
        if letter==unique:
           count+=1
    counts.append((count, unique))

counts.sort()
counts.reverse()

arrpoliteness=[]
arrcount=[]

for i in range(len(counts)):
    count, letter=counts[i]
    #print('%s %d' % (letter,count))
    arrpoliteness.append(letter)
    arrcount.append(count)
#y_pos=np.arange(len(arrpoliteness))
#plt.bar(y_pos,arrcount,align='center',alpha=0.5)
#plt.xticks(y_pos, arrpoliteness)
#plt.ylabel('Frequency')
#plt.title('Frequency of Politeness')
colors=['yellow', 'green', 'red']
plt.pie(arrcount, labels=arrpoliteness, colors=colors, autopct='%1.1f%%')
plt.axis('equal')
plt.show()