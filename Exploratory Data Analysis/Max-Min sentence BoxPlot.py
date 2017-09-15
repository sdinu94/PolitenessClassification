import re
import nltk as nk
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np

def cleaner(s):
    ss=re.sub(r'[^a-zA-Z]', ' ', s) #keeps alpha
    ss2=re.sub("\s\s+" , " ", ss) #removes whitespaces

    tokens = nk.word_tokenize(ss2)
    tokens_lower=map(lambda x:x.lower(),tokens)
    return tokens_lower

df = pd.read_csv(r'C:\Users\sdinu\OneDrive\Documents\02 MS BA\03 Spring 2017\IDS 566 Adv Text Analytics\Project\train.csv')
sentence=df["Request"]

counts=[]
for line in sentence:
    t=cleaner(line)
    counts.append(len(t))
print counts

print max(counts)
print min(counts)
print np.average(counts)
print np.std(counts)

fig = plt.figure()
fig.suptitle('')
ax = fig.add_subplot(111)
plt.boxplot(counts)
plt.show()



