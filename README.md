# master-thesis-ai

This repository consists of code necassary to replicate the experiments carried out in my master thesis "The Effect of Feedback Loops on Fairness in
Rankings from Recommendation Systems" at the Vrije Universiteit Amsterdam.

The repository consists of the following files:
-> Dataset preparation.ipynb - consists of all the data enrichment steps used in the study
-> google_api.py - is used to query Google API for missing author names
-> ol_api.py - is used to query Open Library API for 'works' attribute
-> gender_guesser_api.py - is used to detect gender from names in the dataset
-> data.csv - is a dataset after all pre-processing in a format acceptable by cornac
-> protected - is a text file consisting of book codes relating to books written by female authors
-> experiment.py - is used to run the feedback loops simulation
-> evaluation.py - is used to evaluate all models according to the three main notions of fairness
-> comparison.py - is used to retrieve information about distributions of fairness in 1,5,10 iteration
-> gender_dist_ratings.py - is used to retrieve information about gender distribution in resulting ratings for all models


The public BookCrossing dataset used for this study needs to be separately downloaded from the website of the University of Freiburg, as the size of the dataset was too big to added here.

http://www2.informatik.uni-freiburg.de/~cziegler/BX/ 
