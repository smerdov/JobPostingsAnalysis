In this repo, I analyze data science job postings parsed from websites like Indeed, Dice, and others.
To parse jobs descriptions, I preprocessed words into tokens with ``nltk`` library,
created a count matrix, and reweighted it using TF-IDF.

Representations obtained by TF-IDF were clustered into groups, and it reveals several kinds of DS jobs:

* Big-data engineer (mapreduce, sql, big, query)
* Python machine learning developer (machine, model, analytics)
* Business-oriented data scientist (business, product, analytics)
* Data science manager (manage, communicate, business)
* Data engineer (deploy, engineer, downstream)

