import re
import numpy as np
import nltk as nk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import *
import matplotlib.pyplot as plt
from wordcloud import WordCloud
acro = pd.read_csv(r'C:\Users\sdinu\OneDrive\Documents\02 MS BA\03 Spring 2017\IDS 566 Adv Text Analytics\Project\Text acronym.csv')
text=np.asarray(acro["T"])
actual=np.asarray(acro["A"])
import numpy as np
def cleaner(s):
    ss=re.sub(r'[^a-zA-Z]', ' ', s) #keeps alpha
    ss2=re.sub("\s\s+" , " ", ss) #removes whitespaces

    tokens = nk.word_tokenize(ss2)
    tokens_lower=map(lambda x:x.lower(),tokens) #convert all words to lower
    replaced = []
    for item in tokens_lower:
        words = item
        for i, j in enumerate(text):
            if item == j:
                words = item.replace(item, actual[i])
        replaced.append(words)
    final = []
    for item in replaced:
        j = nk.word_tokenize(item)
        final.append(j)
    qq1 = []
    for qq in final:
        for qq2 in qq:
            qq1.append(qq2)
    filtered_words = [word for word in qq1 if word not in stopwords.words('english')] #remove all stop words
    remove_url = [word for word in filtered_words if not "url" in word] #remove 'url'
    return remove_url
df = pd.read_csv(r'C:\Users\sdinu\OneDrive\Documents\02 MS BA\03 Spring 2017\IDS 566 Adv Text Analytics\Project\train.csv')
sentence=df["Request"]
z=[]
for doc in sentence:
    t = (cleaner(doc))
    for word in t:
        z.append(word)
uniques=[]
for word in z:
    if word not in uniques:
        uniques.append(word)

counts=[]
for unique in uniques:
    count=0
    for word in z:
        if word==unique:
           count+=1
    counts.append((count, unique))
counts.sort()
counts.reverse()

for i in range(min(10,len(counts))):
    count, word=counts[i]
    print('%s %d' % (word,count))
str1=' '.join(z)
wordcloud=WordCloud().generate(str1)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


