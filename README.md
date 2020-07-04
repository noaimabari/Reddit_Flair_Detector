# Reddit_Flair_Detector

This project is a web application made using Python's micro web framework Flask. It is used for detecting the flair of a given reddit post using powerful machine learning algorithms. 
The app can be accessed live from: https://flairdetectorapp.herokuapp.com/. It also contains an end point: /automated_testing which acts like a rest api; it takes a text file(containing reddit post urls in each line) as input in the POST request, detects the flairs of the respective posts and returns the result in a json format. 
For using the endpoint for post request: key='upload_file', value = open('file.txt', 'rb')

r = requests.post("https://flairdetectorapp.herokuapp.com/automated_testing", files = {'upload_file': open('file.txt', 'rb')})

## Codebase
The entire code has been developed using Python programming language, utilizing it's powerful text processing and machine learning modules. The application has been developed using Flask web framework and hosted on Heroku web server.


## Description of files and folders:
The following files and folders are present in the repository:

1. CollectingData.ipynb : Jupyter notebook containing code for collecting data 
2. EDA: Jupter notebook containing code for performing exploratory data analysis on the data.
3. Classifier: Jupyter notebook file containing code for processing data and performing classification of reddit flairs.
4. app.py: file containing the code of the flask web app.
5. model1.pkl: pickled file containing the final saved model.
6. transform1.pkl: pickled file containing the final saved tf idf vectorizer
7. Procfile : Needed to setup heroku
8. requirements.txt: text file containing the python dependencies of the project
9. static: folder containing the css file
10. templates: folder containing the HTML file
11. nltk.txt: nltk dependencies

## Approach:

## 1. Collecting Data
I have first collected data using the reddit api. To ease the process of using the reddit api, I have made use of the praw module which provides various methods for collection of data. Two datasets have been collected containing data for 11 flairs, ie:  **Scheduled, Politics, Photography, Policy/Economy, AskIndia, Sports, Non-Political, Science/Technology, Food, Business/Finance, Coronavirus**.

### Dataset 1: 
232 data points for each of the 11 flair.
Total size: 2552 rows × 9 columns
The first dataset contains the following columns:
1. title: Title of the reddit post
2. No. of comments: Number of comments on the post
3. Created: Time at which post is created
4. id: Id of the post
5. body: Body of the reddit post
6. url: URL given in the post
8. flair: the flair of the post.

### Dataset 2:
232 data points for each of the 11 flairs
Total size: 2552 rows × 2 columns
The second dataset contains:
1. body : Body of the comments
2. flair: the flair of the corresponding post

## 2. Exploratory Data Analysis (EDA)

EDA is performed on the dataset in order to get an overview of the data collected and to summarize its main characteristics.
Feature engineering, removoval of unwanted columns from the dataframe, performing univariate visualisation of its different characteristics, studying of the sentiment polarity of the text, looking at the most occuring words before and after the removal stop words etc. are few of the things explored in this section.

## 3. Classifier

The following steps are being perfomred for processing of data before classification.

1. A data preprocessing step is first performed in order to remove the unwanted symbols and characters.
2. NaN values are removed 
3. Emojis, Stopwords and punctuations are removed.
4. Pos tag of the word is found and lemmatization is performed on the words.
5. Words are converted to lower case.
6. Tf Idf vectorizer is used to transform the data from text to a sparse matrix. Cleaned data now ready.

Following different classifiers are used and experimented with:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Random Forest Classifier
4. Linear Support Vector Machine (SVM)

The following comnbinations are tested with different classifiers:

1. Title:

| Classifier              | Scores        |
| -------------           |:-------------:| 
|Multinomial Naive Bayes| 0.6520376175548589|
|**Logistic Regression**    | **0.6959247648902821**|  
| Random Forest           | 0.6865203761755486|  
| Linear SVM              | 0.6551724137931034|


2. URL:

| Classifier              | Scores        |
| -------------           |:-------------:| 
| Multinomial Naive Bayes|0.4169278996865204|
|**Logistic Regression**    |**0.47648902821316613**|  
| Random Forest           |0.4670846394984326|  
| Linear SVM              |0.4608150470219436|


3. Comments:

| Classifier              | Scores        |
| -------------           |:-------------:| 
|Multinomial Naive Bayes| 0.30139372822299654|
|Logistic Regression    | 0.4146341463414634|  
|**Random Forest**      | **0.44076655052264807**|  
|Linear SVM            | 0.31881533101045295|


4. Title + URL:
 
| Classifier              | Scores        |
| -------------           |:-------------:| 
|Multinomial Naive Bayes|0.603448275862069|
|Logistic Regression    |0.6755485893416928|  
|**Random Forest**      |**0.6755485893416928**|  
|Linear SVM            |0.6551724137931034|


5. Comments + Title :

| Classifier              | Scores        |
| -------------           |:-------------:| 
|Multinomial Naive Bayes|0.44947735191637633|
|Logistic Regression    |0.6533101045296167|  
|**Random Forest**      |**0.7264808362369338**|  
|Linear SVM            |0.5958188153310104|


6. Comments + Title + URL:

| Classifier              | Scores        |
| -------------           |:-------------:| 
|Multinomial Naive Bayes|0.49477351916376305|
|Logistic Regression    |0.6968641114982579|  
|**Random Forest**      |**0.7038327526132404**|  
|Linear SVM            |0.6411149825783972|


From the above results, it can be inferred that the best combination of data and classifier is the use of Random Forest Classifier on Comments + Title combined text data. The score for this combination is 0.7264808362369338. 
