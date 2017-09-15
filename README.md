# PolitenessClassification
Politeness reflects something from inside –an innate sense of consideration for others and respect for self. Politeness is having or showing behavior that is respectful and considerate of other people. Using words such as “thank you” and “please” shows your polite nature. However, politeness has a very wide spectrum, which include your tone the selection of words and the way you say them. The same thing can be said in many ways, but politeness provides you the ability to communicate with others in easy way, which otherwise would have been discourteous.

In this project, I propose a computational framework for identifying linguistic aspects of politeness. In this process, we try to classify a statement as Polite, Neutral or Impolite. I also wanted to compare the performance of multiple Machine Learning algorithms to identify the most suited algorithm for this scenario.

# Data Description

The corpus used in our analysis is based on an online community where requests plays a major role. Politeness is one of the pivotal factors that is obvious when one makes a request. It is a way of expressing your thoughts. We analyzed the requests made by Wikipedia community of editors in the Wikipedia Talk page. The purpose of an article’s talk page is to provide space for editors to discuss changes to its associated article or project page. If someone has a question, concern or comment related to improving an article, they post a note in the article’s talk page. Editor who has a reply to a specific question can comment below it. By this, they can coordinate on the creation of an article. Since these comments are two-way and are consequential conversation, there is a wide room for capturing politeness in this scenario.

There were 29344 requests from Wikipedia, out of which 4353 were annotated. We used 4000 of these 4353 annotated requests for training our model. The annotations are labels that distinguish a request as Polite, Neutral or Impolite. These annotations were made by human where the annotators were made to rate a request on a scale of 5, ranging from very impolite to very polite. The result of their ratings was normalized to yield 3 categories – Polite, Neutral and Impolite. To summarize, the training data consisted of 4000 requests from Wikipedia talks page where each request was classified as Polite, Neutral or Impolite.

# Data Cleaning

The requests made on the page were informal. Hence, they lacked a consistent form. As discussed in data description, in the interest of our current work, there were many inconsistencies among every request made. While few requests were too large that contained lot of unwanted and redundant information, few were very precise and seemed to convey politeness directly. Few consisted of smileys and other text expressions. Hence, it was necessary to clean the data to provide a standard set of information to develop an accurate model.

1.	Removing Acronyms
As discussed earlier, text message acronyms were very predominant in our data. Though acronyms could have been used as such, there were different acronyms that conveyed same meaning. For example, the word ‘Please’ were represented as ‘Plz’ and ‘Pls’. Similarly, there were different forms ok. There were ‘Okay’, ‘Okie’ and ‘OK’. Hence, we need to account for all these combinations into a standard form so that our algorithms could understand what is being conveyed.

A dictionary of around 80 commonly used words that people used while text messaging and their actual English words were created.  It was then imported to Python. Every word in request was compared to the list of text message acronyms in the dictionary and were replaced with standard English words. 

2.	Removing ‘URL’
The data that is being used are all about webpages and articles. Hence, users often cite links to web pages for many reasons. Though these links could have been useful to understand the context of discussion, the data that we obtained contained ‘<url>’ in places where one would expect web links. Since we couldn’t make any sense with just ‘<url>’ and it has been appearing many requests, we tried removing it. This would reduce unimportant information from our data.

3.	Tokenization 
Tokenization is breaking a stream of text into meaningful units. By this, we extract words from a request to understand the meaning it conveys and compare them with other requests under various scenarios. Since our interest at this instance is the words and not the expression, we removed all characters except alphabets using regular expression: ('[^a-zA-Z]', ' ', s). With this, we replace all non-alphabetical characters with a space. The tokenization algorithm in NLTK was used break every sentence in to a list of words. To standardize our approach, we converted all the characters in each string to lower case. This would enable our case sensitive algorithms to treat words equally.

4.	Stop words removal
Stop words are words like: ‘a’, ‘an’, ‘the’, which though gives meaning as we read it as a sentence, they are not of great use for document analysis. It is said that, stop words account for 25% of the words in a document. Hence, removing these stop words would reduce the size of vocabulary significantly. The basic stop words defined in NLTK.CORPUS for English language was used to compare their presence in our document and remove them.

5.	Stemming
Stemming is the process of reducing inflected or derived words to their root form. While bridging the vocabulary gap, it tends to lose precise meaning of each word. Hence, stemming has been limited to certain approach. For the case where we have stemmed the words, Porter Stemmer was used for this purpose.

Note: Out of the methods mentioned above, all the methods weren’t applied to develop all the features used for modeling. Few methods that weren’t suitable to develop a feature were excluded. For example: Tokenization, Stop Words removal, Stemming weren’t applied to cases where we have used POS tagging to extract feature; Tokenization is used only to find TF-IDF vector and for Topic Modeling. 

# Features Extraction

Category 1: Unigrams

The tf-idf vectorizer of sklearn is used to generate the unigrams and their tf-idf count. The term-frequency inverse document frequency metric is used rather than the term frequency to be depict the word in a document as well as in the corpus.

Category 2: Politeness Strategies
All the politeness strategies of ‘A computational approach to politeness with application to social factors’ have been implemented.  

2.1 Features generated using comparisons from pre-existing lists:
-	Hedges
-	Positive
-	Negative
We used the count rather than presence of such strategies.  The idea being the more the number of words the more indicative it is of the class.  It is then normalized.

2.2 Presence of Politeness Strategies
We used regex to identify the patterns in input document instead of the dependency grammar.  The position of the token which was a plus in dependency grammar is made up for by traversing the string to the appropriate token.  The politeness strategies use only positions 1 and 2.

Gratitude: Eg – I really appreciate…
Deference: Eg – Great job…
Greeting: Eg – Hi, how are…
Apologizing: Eg – I am sorry...
Please: Eg – Could you please…
Please Start: Eg – Please do it…	
Indirect (btw): Eg – Ok. By the way…
Direct Question: Eg – What are you…
Direct Start: Eg – So, what…
Counterfactual Modal: Eg – Could you please…
Indicative Modal: Eg – Can you…
1st person start: Eg – I should be…
1st person plural: Eg – Should we do…
1st person: Eg – In my point…
2nd person: Eg – What are you doing…
2nd person start: Eg – You should not…
Factuality: Eg – That is in fact…

Category 3: POS Tags
The NLTK Penn Treebank POS Tagger is used to count the number of occurrences of the following parts of speech.

MD - Modal
Expressions of possibility, suggesting a polite tone.

PRP, PRP$ - Personal and Possessive Pronouns
To capture first and second person accusatory tone in comments. Impersonalization is a measure of politeness. This feature strives to capture the opposite.

JJ, JJR, JJS – Adjective, Adjective comparative, Adjective superlative.  
Adjectives are strong opinion holders.

RB, RBR, RBS – Adverb, Adverb comparative, Adverb superlative
Adverbs used to describe the adjectives

VBD, VBN – Past Tense Verbs
Citing the past in comments could be an indication of politeness.

Category 4: Question Marks
In contexts such as social media where definitive gramma is lacking, symbols are often used to convey as much information as texts.  The use of smileys and emoticons drive home a point that words cannot.  For this reason, we have used question marks as a measure of negativity.
We count the number of question marks used in a sentence.  People tend to use multiple question marks consecutively to drive a point. Also it can be perceived as the more question marks, the ruder a person is.  Polite requests, have question marks but not as many as the impolite ones.  Hence we have considered the count.  It is then normalized.

Union of Features:
The FeatureUnion method of sklearn was used to concatenate all the features.  An advantage of using this method is that the feature generation tasks run in parallel.  The feature generation process is included in Custom Transformers.  These mimic the transformer method of the TFIDF vectorizer.  The Fit method does nothing and hence returns self.
Sample custom transformer:
 

All the features are then fed into a Feature Union –  Notice that the features for which the count is calculated are fed to a Norm() normalization class.

# Machine Learning Algorithms

1.	Baseline Method
Classify all requests as belonging to majority class

The majority class is Neutral with Training data having 2053 out of 4000 samples and the test data having 183 of 353 samples.

The accuracy of the model would then be:
Training – 51.33 %
Testing – 51.84%

We hope for a model that can do better than this naive model
2.	K-nearest neighbors:

K-Nearest Neighbors is a non-parametric model that uses the similarity of the k nearest neighbors to make predictions. 

Modifying 2 parameters: 
-	Number of neighbors to be considered - k
-	Weights – uniform or distance 
o	Default Minkowski calculation used for distance

KNeighborsClassifier() of sklearn was used.

We consider the model that has the highest accuracy on the Validation dataset. Models with K= 12 and 15 have high rates.  We picked the model which also has the highest accuracy on the Training dataset.

3.	Naïve Bayes
Naïve Bayes is a data driven and not a model driven approach.  The data may still have some correlation and so it may not be a suitable method.  Also, it handles purely categorical data well, while we have numeric data in our dataset. It is a generative classifier that models how the data was generated and uses this to classify.

GaussianNB() and CalibratedClassifierCV(method=’sigmoid’) of sklearn were used.  

4.	SVM:
Support Vector Machines allows operating on data as though it were projected onto higher dimensional space, but by using calculations on the original space.  The constant C [values used 0.1, 1.0, 10] sets the tradeoff between maximizing the margin and minimizing the slack.  The Kernels explored were linear, Poly [degrees, 2 and 3] and RBF.  

SVC() of sklearn module was used.

The GridSearchCV() function of sklearn was used to iterate over the all possible values of C, degree and Kernel and the best estimated parameters (which was C=1.0 and kernel = Linear) was used on the Validation dataset.

5.	Maximum Entropy or Logistic Regression or Logit
A logistic regression approach is useful in predicting variables that are categorical. In our case, there are three categories. Hence, this method would suite this scenario. In general, the logistic regression is applicable for cases where the dependent variable is binary. We have three categories in our case. Hence, we have used a multiclass approach to regress our model. 

The sklearn algorithm for Logistic Regression was used. To identify multiple classes, we tweaked the solver parameter to ‘lbfgs’. The default parameter corresponded to classification of binary class. 

# Findings

Comparing all the models, the model with highest accuracy is Logistic Regression.

Training Data: Accuracy = 62.025%
Test Data: Accuracy = 58.64%

The model does better on identifying the polite requests more than the impolite requests.  We can consider adding more features that are indicative of rudeness to further improve the accuracy of the model.
