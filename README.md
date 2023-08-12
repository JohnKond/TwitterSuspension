# Twitter User Suspension Analysis using Graph Embeddings


## Author

- [@Ioannis Kontogiorgakis](https://github.com/JohnKond)
- [@Alexander Shevtsov](https://github.com/alexdrk14)


## Abstract

On February 2022 Russia invaded Ukraine, causing the well-known Russo-Ukraine war. This act directly triggered turmoil on social media platforms, since users all over the world debate and offer their opinion on this matter. Posts on Twitter, are often accompanied by hashtags relevant to the content of the post. Throughout this period, from February to June, in which war was at its climax, we collected data , originated from 8 million users. Exploiting Twitter open API, we extracted the social relations between users in graph representations.
In this paper, we seek to answer the question **"Is it possible to predict a Twitter user account suspension, based solely on social relations of the user ?"**. We present a machine learning pipeline, in which we analyze how well a machine learning model can predict user suspension, and how the accuracy of the model changes during the period February to June.



## Features
- **Graph Representation:** Build a graph representation of Twitter users and their interactions.
- **Graph Embeddings:** Utilize graph embedding algorithms to learn vector representations of users in the graph.
- **Data Processing:** Preprocess raw Twitter data to extract meaningful interaction features.
- **Suspension Analysis:** Analyze embedding patterns to identify potential factors contributing to user suspensions.
- **Suspension Prediction:** Utilize machine learning algorithms in order to classify suspended and non suspended users.


 
 
## Graph Embeddings

We transform the multi-layered social graph from Twitter into graph embeddings to make it compatible with machine learning models. After exploring conventional graph embedding techniques like DeepWalk, LINE, Node2Vec, SDNE, and Struc2Vec, we turned to Facebook's **Pytorch-BigGraph**, a Neural Network solution for accurate representations. The chosen approach involves training the model to create numerical vectors for users, where we found that an output vector dimension of 150 offers optimal performance based on MRR and AUC scores.

## Data Pre-processing
- **Dataset Balance:** To create a balanced dataset and mitigate training bias, we used random under-sampling, equalizing the number of suspended and normal users.
- **Feature Selection:** To prevent overfitting, we employed Lasso regression to select the most relevant features. We fine-tuned the hyper-parameter alpha through cross-validation, then pruned low-significance features.
- **Scaling**: Data normalization was achieved with the MinMax scaler, confining user embeddings to a 0-1 range, aiding the model's grasp of the user suspension problem and averting overgeneralization issues.


## Model Selection

In pursuit of predicting user account suspension accuracy based on social relationships, we experimented with four models :
- **XGBoost**
- **Random Forest**
- **Linear Regression**
- **Naive Bayes**

in order to determine the most effective classification approach. Using the first-month graph dataset (February - March), we divided our input data into training and validation sets (80/20 ratio), leveraging K-fold cross-validation to evaluate each model's performance across five folds. Upon hyper-parameter tuning and analysis, XGBoost emerged as the best-performing model according to validation scores. This model, along with its optimized parameters, was subsequently employed for final training on the entire first-month dataset.

## Evaluation

- **1st Scenario :** We aimed to evaluate the model's adaptation to the initial month's data (February-March). To achieve this, we trained the model on the complete first-month graph dataset, using both the train and test sets.The model's initial performance was assessed by predicting user suspensions in subsequent one-month periods: March-April, April-May, and May-June.
- **2nd Scenarion :** This scenario delved into the evolving nature of user embeddings over months, particularly during the significant surge in user numbers due to geopolitical events. To gauge improvement over time, the model was trained monthly with embeddings from the February-March dataset. The goal was to ascertain if the model's performance enhanced as it learned from the progressively evolving embeddings. Each month's user embeddings were trained on the February-March dataset, which held a larger user count due to the Russo-Ukraine war's peak.
- **Third Scenario:** Our final scenario sought to measure the model's accuracy across all graph embedding datasets over the four-month period. To ensure fairness, we balanced datasets using random undersampling and split them into train and test sets (80/20 ratio). Following normalization with Min-Max scaler, we loaded the scaler for test set normalization. We aimed to comprehend how accuracy fluctuated across the dataset dimensions, accounting for each graph containing the previous ones.

<!--
## Installation

- Clone the repository :
```git clone https://github.com/JohnKond/TwitterSuspension.git```

- Install essential requirements, by navigating in project folder and run :
```pip install -r requirements```

- Export graphs from database, using pymongo
```python graphUtils/export_graphs.py --period feb_mar ```
    > run this task for each period (feb_mar, feb_apr, feb_may, feb_jun)

- Graph Embeddings 
After graph export, lets assume that your graph files are stored in this format in folder data: 

```
├── data
│   ├── feb_mar
│   │   ├── graph_mention_feb_mar.tsv
│   │   ├── graph_quote_feb_mar.tsv
│   │   ├── graph_multy_feb_mar.tsv
│   ├── feb_apr
│   │   ├── ...
│   ├── feb_may
│   │   ├── ...
│   ├── feb_jun
│   │   ├── ...
```
--!>
