# Gender-Classification-Challenge

The code in this repo is for the Gender Classification challenge for the video 'Learn Python for Data Science #1' by @Sirajology (Siraj Raval) on YouTube. Although not an official submission (I was late to discover these awesome challenges), this is my attempt at it. We us the scikit-learn machine learning library to train a decision tree on a small dataset of body metrics (height, width, and shoe size) which are labeled based on gender. We then predict the gender of someone given a test set of body metrics. 

# Dependencies

* [numpy](http://www.numpy.org/)
* [Scikit-learn](http://scikit-learn.org/stable/install.html)
* [scipy](https://www.scipy.org/)

Install the missing dependencies using pip
~~~~
pip install numpy scikit-learn scipy
~~~~

# Using the project

After installing the dependencies, run the script in terminal 

~~~~
python Gender_Classifer.py
~~~~

Based on the accuracy results, the best classifier here is Naive Bayes.


# Credits

The boilerplate code and the test data for this challenge was provided by [Siraj Raval] (https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A).

