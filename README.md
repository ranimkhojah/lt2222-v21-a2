# LT2222 V21 Assignment 2

Assignment 2 from the course LT2222 in the University of Gothenburg's winter 2021 semester.

Name: *Ranim Khojah*


## High-level Documenation


### Part 1- Preprocessing
The purpose of the preprocessing stage was to i) remove uninformative data (dirty) data that may make the model a bit bias and ii) reduce the size of the data. So, to achive that, the words in dataset were lemmatized and lowercased, then stopwords and punctuation were filtered out. and finally, the new 'processed' words replaced the old words and unnessary rows were eliminated.


### Part 2- Creating instances
When creating instances, few design decisions has been made. For instance, the context/ features of a NE instance does **not** include other named entities. This is because the range of possible NEs that can be in a context are huge, e.g., "The _Swedish_ artist likes to paint the streets of _Paris_" these two NEs can be substituted with so many NEs yet stay 100% valid. So if the decision was to include NEs in the context, then it would probably be best to train another model specified to recognize proper nouns and named entities.
Moreover, to handle special scenarios where the NE occur at the beginning, at the end or even close to them, start and end tokens were incorporated into the features array. After testing and evaluating the model later, it was noted that these tokens enhanced the performance of the model by increasing the accuracy.

### Part 3- Reducing and splitting
To convert the text-data to a numeric format that would be better concieved by the machine, dimensionality reduction was performed using truncated singular value decomposition (SVD). To get the best dimensions that the matrix should be reduced to, few experiments were conducted using different dimention ranges e.g., dims=1000, ... dims=500 and dims=400. It was decided later to go with dims=490 as it resulted in a better accuracy of the model since the dimentions that are lower that 490 resulted in an effect similar to underfitting and vice versa i.e. overfitting to dimentions greater than 700.

### Part 5- Evaluation
The confusion matrix of the predictions based on the training data were obviously significantly better than the testing data, since the former is evaluated based on the same data that it was trained it. That is similar to when a teacher tests the students with an exam that is an exact copy of an old mock exam that the students studied from.  
In general, the training data's confusion matrix report around ~100%-98% accuracy depending on the random splits of the data. Whereas the testing data's matrix reports more False positives.

### Bonus A
The two classes that fail the most during the testing of the model are Geographical Entities i.e. geo (confused with Geo-political Entities i.e. gpe) and art. The former can fail due to the similarity of the contexts that both NEs (with geo and gpe classes) can have. For instance, the geo Pakistan NE and the gpe Pakistani NE have the following contexts (The highlighted words are the common features):
pakistani - Sent. no. 52.0 - Context: ('</S1>', '</S2>', '**senior**' , '**military**', '**official**', '**say**', '**put**', '**call**')
pakistan - Sent. no. 52.0 - Context: ('**senior**', '**military**', '**official**', '**say**', '**put**', '**call**', '""""', 'sordid', 'chapter', '""""')

On the other hand, the size of the training data can also affect the accuracy of the prediction. Therefore, the predictions of NEs with art class usually fail since the training data doesn't have many data points with labeled with art class.


### Bonus B
The POS tags were incorporated in the features vector by concatenating each word in the context with its corresponding POS tag following the format WORD-POS e.g., pregnancy-NN and pregnant-JJ. This way, the vocabulary that contains the unique features of all NEs will be larger since for example "park-NN" and "park-VB" will be considered as two different features, which was not the case when POS tags were not considered. Although not significantly, this process increased the accuracy of the model (judging by the confusion matrix of the model evaluation).



*Note: the jupyter notebook is the result of executing 20,000 out of 60,000 data points due to limited resources*
