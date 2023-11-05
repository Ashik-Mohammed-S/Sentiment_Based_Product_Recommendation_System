## Sentiment-Based Product Recommendation

### Problem Statement
E-commerce has revolutionized the way we shop. Companies like Amazon, Flipkart, Myntra, Paytm, and Snapdeal have made it possible to order products from the comfort of our homes. Imagine you're a Machine Learning Engineer at an e-commerce company called 'Ebuss'. Ebuss has a diverse product range, including household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products, and health care products. To compete with market leaders like Amazon and Flipkart, Ebuss needs to leverage technology and grow rapidly in the e-commerce market.

### Solution Approach

- **Data Preparation**: The dataset and attribute descriptions are available under the dataset folder. The data is cleaned, visualized, and preprocessed using Natural Language Processing (NLP).

- **Text Vectorization**: The TF-IDF Vectorizer is used to vectorize the textual data (review_title+review_text). This measures the relative importance of a word with respect to other documents.

- **Handling Class Imbalance**: The dataset suffers from a class imbalance issue. The Synthetic Minority Over-sampling Technique (SMOTE) is used for oversampling before applying the model.

- **Machine Learning Models**: Various Machine Learning Classification Models are applied on the vectorized data and the target column (user_sentiment). These models include Logistic Regression, Naive Bayes, and Tree Algorithms (Decision Tree, Random Forest, XGBoost). The objective of these ML models is to classify the sentiment as positive(1) or negative(0). The best model is selected based on various ML classification metrics (Accuracy, Precision, Recall, F1 Score, AUC). XGBoost is selected as the best model based on these evaluation metrics.

- **Recommender System**: A Collaborative Filtering Recommender system is created based on User-user and Item-item approaches. The Root Mean Square Error (RMSE) evaluation metric is used for evaluation.

- **Code**: The code for Sentiment Classification and Recommender Systems is available in the Main.ipynb Jupyter notebook.

- **Product Filtering**: The top 20 products are filtered using the recommender system. For each of these products, the user_sentiment is predicted for all the reviews. The Top 5 products with the highest Positive User Sentiment are then filtered out.