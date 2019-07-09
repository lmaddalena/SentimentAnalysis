# Sentiment Analysis

Predicting if a customer's review has positive or negative sentiment using the Wikipedia detox dataset <https://raw.githubusercontent.com/dotnet/machinelearning/master/test/data/wikipedia-detox-250-line-data.tsv>.
The first column represents the sentiment of the text (0 is non-toxic, 1 is toxic), and the second column represents the comment left by the user. The columns are separated by tabs.

```text
Sentiment Analysis
======================

Loading data....
Processing data....
Data file contains 250 comments

Generating Vocabulary List....
Vocabulary generated with 303 words

Retrieving labels...
Extracting features...
Splitting data...

Test set: 50
Training set: 200

Training linear SVM ...

....*
optimization finished, #iter = 847
nu = 0.316584072818237
obj = -16.11023335973633, rho = -0.5253018259713941
nSV = 131, nBSV = 23
Total nSV = 131

Training set Accuracy: 97.00000

Test set Accuracy: 70.00000
```
