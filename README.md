#  Sentiment Analysis on Amazon Reviews with Online Learning

This is a project on Sentiment analysis of Amazon product reviews where each review will be classified as positive or negative. Once the classification step is done, an interactive website will be published where users will be prompted to enter a review for a product. The algorithm we developed will be classifying this newly entered review as positive or negative, based on the previous learnings. The user will then be able to give feedback whether the model classified the review correctly or not and based on this the model will be trained further with the new example, thus mimicking the out-of-core learning.

I have used Python for the most part of my project, Flask and HTML along with Python for creating my web app. I have hosted my web app on PythonAnywhere and can be accessed here: <http://srisowmya.pythonanywhere.com/>.

## Organization

You can access the codes for both my sub projects from SentimentClassifier.ipynb and RatingPredictor.ipynb. The dataset I used is downloaded from Kaggle in a CSV format and can be accessed from the main thread. The website folder consists of all the files required to host my website with the pickle objects, HTML templates, CSS sheet and the app.py file. 

## Data Preparation

I have removed all the unnecessary columns from the dataset which won't contribute to the sentiment analysis. I have retained only the review text as well as the review rating given by the customers. Based on the exploration, I have found that the positive reviews are way too high when compared to the negative ones. This is something which I observed personally too that on amazon there are usually around 85-90% positive reviews when compared to the negative ones which vary from 5-10%. An exception is of course if the product is very bad. But, most products do have a high volume of positive reviews. I have down sampled the positive reviews in order to avoid any biasing during the training process. The code used for this process can be found in my jupyter notebooks.

## Training and Evaluation
First, I have created tokenizer methods to create tokens from the reviews and used [TFIDF vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) for creating the inverse document frequency.

For the sentiment and rating classification, I have used the [SGD Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) which is short for Stochastic Gradient Descent Classifier. This classifier allows the mini batch training which is required for the online learning process. 

I have performed a grid search with a given set of parameters to get the best model based on accuracy. I have performed this by creating a pipeline for the vectorizer and classifier.

After getting the best estimator parameters, I have created a HashingVectorizer with the decided parameters and pickled the classifier object. I have also saved a python file with the chosen estimator, vectorizer and the tokenizer.

After saving the file which are required for the web hosting, I have created a sql lite database to store the reviews entered from the web app i.e. the online reviews. 

## Web Application
I have used Flask to prepare my web application and also created templates for each HTML webpage. You can access the code for my application from app.py. Here, I am loading the pickled classifier and reading the review entered by the customer. I then call the tokenizer and the vectorizer methods written separately in the saved python file and run my classifier on it. Then, I display the prediction result and take the customer's feedback in order to train the classifier based on the correct results.

I have first developed this locally and once, I have finished developing the application, I created a free account in PythonAnywhere and uploaded the files there to host the application. 

As you can see, I display the Home screen first giving a quick introduction on how to navigate the website. The 'Sentiment and Rating Predictor' tab is where the customer will be giving the product review. After clicking on Submit Review, I display the results of both the models and the user can provide feedback by validating the results shown. If the results are incorrect, the model gets trained on whatever the user chooses. Therefore, the user helps the model to be exposed to various reviews and learn the correct classification.

The 'Sentiment Analysis' tab loads the data of all the different reviews given by the user and shows the correct sentiment in there. For making it more interesting, I have included the prediction given by the model too. This way the user gets a chance to test if the model is learning incrementally or not. Similarly, I have also included a 'Rating Analysis' tab, to show the results from the rating predictions.

## Conclusion
Once the model has been trained, it keeps getting better and continues to learn from the online web application. When the user enters a review through the web application, there is an option to submit feedback based on whether the machine prediction is correct or incorrect. This feedback goes back to the system and it becomes a part of the learning. If the system has predicted something incorrectly, the user can submit what the actual rating/sentiment should be and the system learns from this and predicts the rating or sentiment correctly the next time. It may take a few iterations for the model to learn the correct prediction. Feel free to send feedback or suggestions!

## References
1. <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html>
2. <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html>
3. <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>
4. <https://www.datacamp.com/community/tutorials/stemming-lemmatization-python>
5. <https://flask.palletsprojects.com/en/1.1.x/quickstart/>
6. <https://blog.pythonanywhere.com/121/>
7. <https://www.sqlitetutorial.net/>
