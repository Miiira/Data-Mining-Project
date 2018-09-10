This is a data mining project I did independently at Tencent in Summer 2018. I built a classification model for a bank to distinguish the potential customers who will buy financial products.

From the original 100GB of financial data, I filtered out the data of more than 30000 corporates due to the memory of local server. The data include basic information, assets, purchase activities of corporates.

In the python script, I used libraries such as numpy, pandas, and scikit learn to do feature engineering, data visualization, and build machine learning models. I built and selected about 18 features and converted them to numeric features. Because the original data is imbalanced, I used SMOTE to do oversampling. 

Then I trained the model using algorithms such as XGBoost, Random Forest, SVM, Neural Network, and used accuracy, precision, recall, and f2 score to evaluate the models. Then I picked several basic models to do ensemble modeling and finally got a model with 97.7% accuracy, 95% precision, and 84% recall rates.