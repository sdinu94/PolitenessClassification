
# coding: utf-8

# In[ ]:

#Libraries used in the Project


# In[7]:

import pandas
import numpy
import nltk
import re
import sklearn
from sklearn.pipeline import Pipeline,TransformerMixin,FeatureUnion
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import cross_val_predict
from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn import preprocessing
from sklearn import decomposition
from nltk.corpus import stopwords, treebank
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC


# In[3]:

#Read input file
df = pandas.read_csv("/Users/Vijitha/Desktop/Spring/IDS 566/wikipedia.annotated_train-2.csv")

#Read positive and Negative Dictionaries
pos=pandas.read_csv("/Users/Vijitha/Desktop/Spring/IDS 566/Positive words.csv")
positive=pos["Words"]
neg=pandas.read_csv("/Users/Vijitha/Desktop/Spring/IDS 566/Negative words.csv")
negative=neg["Words"]

#Read Dependent variable
Y=df['politeness'].as_matrix()

#Read Independent Variable
X=df['Request']

# Word list for Hedges politeness strategy
hedges = [
    "think", "thought", "thinking", "almost",
    "apparent", "apparently", "appear", "appeared", "appears", "approximately", "around",
    "assume", "assumed", "certain amount", "certain extent", "certain level", "claim",
    "claimed", "doubt", "doubtful", "essentially", "estimate",
    "estimated", "feel", "felt", "frequently", "from our perspective", "generally", "guess",
    "in general", "in most cases", "in most instances", "in our view", "indicate", "indicated",
    "largely", "likely", "mainly", "may", "maybe", "might", "mostly", "often", "on the whole",
    "ought", "perhaps", "plausible", "plausibly", "possible", "possibly", "postulate",
    "postulated", "presumable", "probable", "probably", "relatively", "roughly", "seems",
    "should", "sometimes", "somewhat", "suggest", "suggested", "suppose", "suspect", "tend to",
    "tends to", "typical", "typically", "uncertain", "uncertainly", "unclear", "unclearly",
    "unlikely", "usually", "broadly", "tended to", "presumably", "suggests",
    "from this perspective", "from my perspective", "in my view", "in this view", "in our opinion",
    "in my opinion", "to my knowledge", "fairly", "quite", "rather", "argue", "argues", "argued",
    "claims", "feels", "indicates", "supposed", "supposes", "suspects", "postulates"
]

#Read the acronym_expansion file for wikipedia
acro = pandas.read_csv(r"/Users/Vijitha/Desktop/Spring/IDS 566/Text acronym.csv")
text=numpy.asarray(acro["T"])
actual=numpy.asarray(acro["A"])


# In[4]:

#Cleaner function
def cleaner(s):
    ss=re.sub(r'[^a-zA-Z]', ' ', s) #keeps alpha

    tokens = nltk.word_tokenize(ss)
    tokens_lower=map(lambda x:x.lower(),tokens) #convert all words to lower
    replaced = []
    for item in tokens_lower:
        words = item
        for i, j in enumerate(text):
            if item == j:
                words = item.replace(item, actual[i])
        replaced.append(words)
    filtered_words = [word for word in replaced if word not in stopwords.words('english')] #remove all stop words
    remove_url = [word for word in filtered_words if not "url" in word] #remove 'url'
    return remove_url

#TFIDF on unigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 1),token_pattern=r'\b\w+\b', min_df=1)

#QMarks count
class QMarks(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda s: s.count('?')))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#QMarks atStart
class StartQ(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda s: s.startswith('?')))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#Normalize values
class Norm(TransformerMixin):

    def transform(self, X, **transform_params):
        return DataFrame(preprocessing.normalize(X))

    def fit(self, X, y=None, **fit_params):
        return self


# In[ ]:

#Strategy: Hedges
class Hedgeser(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda l: len(set(nltk.word_tokenize(l.lower())).intersection(hedges))))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#Strategy: Positive lexicon
class Poser(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda l: len(set(nltk.word_tokenize(l.lower())).intersection(set(positive)))))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#Strategy: Negative Lexicon
class Neger(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda l: len(set(nltk.word_tokenize(l.lower())).intersection(set(negative)))))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

    #Strategy: Counterfactual Modals
class SubjunctiveTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda s: "could you" in s.lower() or "would you" in s.lower()))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#Strategy: Indicative Modal
class IndicativeTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda s: "can you" in s.lower() or "will you" in s.lower()))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#Strategy: Gratitude
class Gratitude(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda s: "i appreciate" in s.lower() or "thank" in s.lower()))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#Strategy: Deference
class Deference(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda s: s.partition(' ')[0].lower() in ["great","good","nice","good","interesting","cool","excellent","awesome"]))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#Strategy: Greeting
class Greeting(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda s: s.partition(' ')[0].lower() in ["hi","hello","hey"]))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#Strategy: DirectStart
class DirectStart(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda s: s.partition(' ')[0].lower() in ["so","then","and","but","or"]))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#Strategy: First person start
class FirstpStart(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda s: s.partition(' ')[0].lower() in ["i","my","mine","myself"]))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#Strategy: First person plural
class FirstpPlural(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda l: len(set(nltk.word_tokenize(l.lower())).intersection(["we", "our", "us", "ourselves"]))>0))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#Strategy: Second Person
class SecondPerson(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda l: (len(set(nltk.word_tokenize(l.lower())).intersection(["you","your","yours","yourself"]))>0)and (l.partition(' ')[0].lower() not in ["you","your","yours","yourself"])))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#Strategy: Second person start
class SecondpStart(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda s: s.partition(' ')[0].lower() in ["you","your","yours","yourself"]))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#Strategy: PleaseStart
class PleaseStart(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda s: s.partition(' ')[0].lower() in ["please"]))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#Strategy: FirstPerson
class FirstPerson(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda l: (len(set(nltk.word_tokenize(l.lower())).intersection(["i", "my", "mine", "myself"]))>0)and (l.partition(' ')[0].lower() not in ["i", "my", "mine", "myself"])))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#Strategy: Please
class Please(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda l: (len(set(nltk.word_tokenize(l.lower())).intersection(["please"]))>0)and (l.partition(' ')[0].lower() not in ["please"])))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#Strategy: Factuality
class Factuality(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda l: (len(set(nltk.word_tokenize(l.lower())).intersection(["really", "actually", "honestly", "surely"]))>0) or "the point" in l.lower() or "the reality" in l.lower() or "the truth" in l.lower() or "in fact" in l.lower()))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#Strategy: Direct Question
class Question(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda s: s.partition(' ')[0].lower() in ["please"]))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#Strategy: Indirect (btw)
class Bytheway(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda s: "by the way" in s.lower()))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#Strategy: Apologizing
class Apologizing(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda s: len(set(nltk.word_tokenize(s.lower())).intersection(["sorry", "whoops","oops","excuse", "regret", "admit","plea",]))>0 or "i apologize" in s.lower() or "forgive me" in s.lower() or "excuse me" in s.lower()))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self


# In[ ]:


#POSTags: Modal
class POSTaggerMD(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda s: sum(1 for x in dict(nltk.pos_tag(s)).values() if x=='MD')))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#POSTags: Pronouns
class POSTaggerPRP(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda s: sum(1 for x in dict(nltk.pos_tag(s)).values() if x=='PRP' or x=='PRP$')))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#POSTags: Adverbs
class POSTaggerWD(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda s: sum(1 for x in dict(nltk.pos_tag(s)).values() if x=='RB' or x=='RBR' or x=='RBS')))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#POSTags: Adjectives
class POSTaggerJJ(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda s: sum(1 for x in dict(nltk.pos_tag(s)).values() if x=='JJ' or x=='JJR' or x=='JJS')))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self

#POSTags: Past verbs
class POSTaggerVB(TransformerMixin):

    def transform(self, X, **transform_params):
        modals = DataFrame(X.apply(lambda s: sum(1 for x in dict(nltk.pos_tag(s)).values() if x=='VBD' or x=='VBN')))
        return modals

    def fit(self, X, y=None, **fit_params):
        return self




# In[5]:

#Unsupervised Topic Modeling
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])


# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, tokenizer=cleaner)
tf = tf_vectorizer.fit_transform(X)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 3

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0)
lda.fit(tf)

print("\nTopics in LDA model:")
display_topics(lda, tf_feature_names, 50)


# In[8]:

#Generic Model Fit if we want to use multiple models in FeatureUnion
class ModelTransformer(TransformerMixin):

    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))

#Combine features to run in parallel
combined_features= FeatureUnion([("tfd",vectorizer),
                                ("pos",SubjunctiveTransformer()),
                                ("pos1",IndicativeTransformer()),
                                ("try", Pipeline([
                                    ('modal',POSTaggerMD()),
                                    ('scale', Norm())
                                    ])),
                                ("try1", Pipeline([
                                    ('pro',POSTaggerPRP()),
                                    ('scale', Norm())
                                    ])),
                                ("try2", Pipeline([
                                    ('wd',POSTaggerWD()),
                                    ('scale', Norm())
                                    ])),
                                ("try3", Pipeline([
                                    ('adj',POSTaggerJJ()),
                                    ('scale', Norm())
                                    ])),
                                ("try4", Pipeline([
                                    ('verb',POSTaggerVB()),
                                    ('scale', Norm())
                                    ])),
                                ("hedges", Pipeline([
                                    ('hed',Hedgeser()),
                                    ('scale',Norm())
                                    ])),
                                ("positive",Pipeline([
                                    ('posi',Poser()),
                                    ('scale',Norm())
                                    ])),
                                ("negative",Pipeline([
                                    ('nega',Neger()),
                                    ('scale',Norm())
                                    ])),
                                ("qmarks",Pipeline([
                                    ('qm',QMarks()),
                                    ('scale',Norm())
                                    ])),
                                ("gratitude",Gratitude()),
                                ("startq",StartQ()),
                                ("deference",Deference()),
                                ("greeting", Greeting()),
                                ("dstart", DirectStart()),
                                ("firststart", FirstpStart()),
                                ("firstplu", FirstpPlural()),
                                ("secper", SecondPerson()),
                                ("secpstart", SecondpStart()),
                                ("pls", PleaseStart()),
                                ("fp", FirstPerson()),
                                ("please", Please()),
                                ("fact", Factuality()),
                                ("qs", Question()),
                                ("btw", Bytheway()),
                                ("sorry", Apologizing())
                                ])


# In[9]:

#Fit training data
X_features = combined_features.fit(X, Y).transform(X)
X_features.toarray()

#Read the test data
dftest = pandas.read_csv("/Users/Vijitha/Desktop/Spring/IDS 566/wikipedia.annotated_test.csv")

Y1=dftest['Label'].as_matrix()

X1=dftest['Request']

#Transform test data
Test_features=combined_features.transform(X1)
print "features generated"

#Final Model
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X_features.toarray(),Y)

#Cross validate on Test and Train
pred=cross_val_predict(clf, Test_features.toarray(), Y1, cv=3)
predtr = cross_val_predict(clf,X_features.toarray(),Y,cv=3)


# In[10]:

#Print Metrics of Classifier
print "train"
print(metrics.classification_report(Y, predtr))
print(metrics.confusion_matrix(Y,predtr))
print "score"
print metrics.accuracy_score(Y, predtr)

print "test"
print(metrics.classification_report(Y1, pred))
print(metrics.confusion_matrix(Y1,pred))
print "score"
print metrics.accuracy_score(Y1, pred)


# In[ ]:

#SVM implementation to find best parameters
svm = SVC()

# Do grid search over kernel, degree and C:

pipeline = Pipeline([("svm", svm)])

param_grid = dict(svm__kernel=["linear","poly","rbf"],
                  svm__degree=[2,3],
                  svm__C=[0.1, 1, 10])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
grid_search.fit(X_features.toarray(), Y)
print(grid_search.best_estimator_)


# In[ ]:

#Naive Bayes Implementation

clf = GaussianNB()
clf.fit(X_features.toarray(), Y)

pred = cross_val_predict(clf, Test_features.toarray(), Y1, cv=5)
predtr = cross_val_predict(clf, X_features.toarray(), Y, cv=5)
print "GaussianNB"
print "train"
print(metrics.classification_report(Y, predtr))
print(metrics.confusion_matrix(Y, predtr))
print "score"
print metrics.accuracy_score(Y, predtr)
print "GaussianNB"
print "test"
print(metrics.classification_report(Y1, pred))
print(metrics.confusion_matrix(Y1, pred))
print "score"
print metrics.accuracy_score(Y1, pred)

clf = CalibratedClassifierCV(clf, method='sigmoid')
clf.fit(X_features.toarray(), Y)

pred = cross_val_predict(clf, Test_features.toarray(), Y1, cv=5)
predtr = cross_val_predict(clf, X_features.toarray(), Y, cv=5)
print "Sigmoid"
print "train"
print(metrics.classification_report(Y, predtr))
print(metrics.confusion_matrix(Y, predtr))
print "score"
print metrics.accuracy_score(Y, predtr)
print "Sigmoid"
print "test"
print(metrics.classification_report(Y1, pred))
print(metrics.confusion_matrix(Y1, pred))
print "score"
print metrics.accuracy_score(Y1, pred)


# In[ ]:

#knn implementation for various values of k 
#iterate for different values of k
for k in [12,20,18,3,30,25,7]:
    clf = KNeighborsClassifier(k,weights='distance')
    clf.fit(X_features.toarray(),Y)

    pred=cross_val_predict(clf, Test_features.toarray(), Y1, cv=3)
    predtr = cross_val_predict(clf,X_features.toarray(),Y,cv=3)
    print k
    print "train"
    print(metrics.classification_report(Y, predtr))
    print(metrics.confusion_matrix(Y,predtr))
    print "score"
    print metrics.accuracy_score(Y, predtr)
    print k
    print "test"
    print(metrics.classification_report(Y1, pred))
    print(metrics.confusion_matrix(Y1,pred))
    print "score"
    print metrics.accuracy_score(Y1, pred)


# In[ ]:

#logit implementation
logreg = linear_model.LogisticRegression(C=2, solver='lbfgs')
logreg.fit(X_features, Y)

pred=cross_val_predict(clf, Test_features.toarray(), Y1, cv=5)
predtr = cross_val_predict(clf,X_features.toarray(),Y,cv=5)

print "train"
print(metrics.classification_report(Y, predtr))
print(metrics.confusion_matrix(Y,predtr))
print "score"
print metrics.accuracy_score(Y, predtr)
print "test"
print(metrics.classification_report(Y1, pred))
print(metrics.confusion_matrix(Y1,pred))
print "score"
print metrics.accuracy_score(Y1, pred)

