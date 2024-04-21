
# PRML Project
# Exploring Sentiment Analysis on Twitter Dataset

In this project, we've harnessed traditional machine learning algorithms including **SVM, ANN, Naive Bayes, Decision Trees, Random Forest, and Logistic Regression** to discern sentiment in Twitter comments.giOur approach encompasses two key features: **random text analysis** and the identification of **abusive users sourced from CSV files**.

## Run Locally

Clone the project

```bash
  git clone https://github.com/AnujPatil110377/Prml_project/
```



## Authors

- [@Anuj_Vijay_Patil](https://github.com/AnujPatil110377)
- [@Rohan_lambture](https://github.com/RohanLambture)
- [@Sachin_raj](https://github.com/RohanLambture)
- [@Ritesh_fageria](https://github.com/riteshfageriya)
- [@Yashraj](https://github.com/yshrjnpm)


## Work Distribution

- Implementation of Models
   - **Sachin**: 
        - Svm
        - Project Website
    - **Rohan**:
        - Logistic Regression
        - Random Forest
        - Report
    - **Anuj**:
        - Aritficial Neural Network
        - Git Repository Handling
        - Model Deployment on Streamlit
    - **YashRaj**:
        - Naive Bayes 
        - Model Integration into one script
        - Report
    - **Ritesh**:
        - Decision Tree
        - Readme file




## Screenshots

![App Screenshot](https://raw.githubusercontent.com/AnujPatil110377/Prml_project/main/Home_page.jpg)



## Demo

**Model Hosted On streamlit.io Platform**
   - https://prmlproject-pfevkhw8iirkw3rythzv2c.streamlit.app/
**Usage Instructions**:
 - Csv Analysis-  Make sure your **".Csv"** file has atleast two columns (**Name**,**Tweet**)


## Features

- **Text Analysis**
   - ```bash
     test.py # Contains Model integration to  Predict text on all models at once
     ```
- **Csv Analysis**
   - ```bash
     Spam_user_blacklist.py.py # Contains Model integration to Predict Csv data Identify abusive user with negative count >3
     ```
 


## Documentation

[Report](https://github.com/AnujPatil110377/Prml_project/blob/main/Prml_project_report.pdf)


## Optimizations

- **Implemented varity of Data preprocessing method for data cleaning :**
    - Tokenization 
    - Stop word Removal
    - Sign,Html,Punctuation Removal
    - Vectorization
- **Used Grid Search and Iterative method for hyper Parameter Tuning**
    - We achieved impressive accuracies with **logistic regression** and **Decision trees surpassing** 80%, while **ANN** and **Naive Bayes** reached peaks of 70%

