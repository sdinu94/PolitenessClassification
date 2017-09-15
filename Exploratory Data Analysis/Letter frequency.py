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
    tokens_lower=map(lambda x:x.lower(),tokens) #convert all words to lower
    #filtered_words = [word for word in tokens_lower if word not in stopwords.words('english')] #remove all stop words
    remove_url = [word for word in tokens_lower if not "url" in word] #remove 'url'
    return remove_url
df = pd.read_csv(r'C:\Users\sdinu\OneDrive\Documents\02 MS BA\03 Spring 2017\IDS 566 Adv Text Analytics\Project\train.csv')
sentence=df["Request"]
z=[]
for doc in sentence:
    t = (cleaner(doc))
    for word in t:
        for letter in word:
            z.append(letter)

uniques=[]
for word in z:
    if word not in uniques:
        uniques.append(word)

counts=[]
for unique in uniques:
    count=0
    for letter in z:
        if letter==unique:
           count+=1
    counts.append((count, unique))
counts.sort()
counts.reverse()
arrletter=[]
arrcount=[]
for i in range(len(counts)):
    count, letter=counts[i]
    #print('%s %d' % (letter,count))
    arrletter.append(letter)
    arrcount.append(count)
y_pos=np.arange(len(arrletter))
plt.barh(y_pos,arrcount,align='center',alpha=0.5)
plt.yticks(y_pos,arrletter)
plt.xlabel('Frequency')
plt.title('Frequency of letters')
plt.show()
