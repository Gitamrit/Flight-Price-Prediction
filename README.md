# Flight Price Prediction Web Application

The Flight Price Prediction Web Application is designed to assist users in estimating flight prices based on various input parameters. Key features include:

+ User Inputs:
  + Users provide details such as airline, source, destination, departure time, arrival time, and duration.
  + These inputs serve as the basis for predicting flight prices.
+ Frontend Technologies:
  + The application is built using Flask for the backend logic.
  + HTML and CSS are used for creating the frontend interface.
+ Backend:
  +  A machine learning model is trained on flight price prediction dataset avalaible on Kaggle to predict flight prices.
  +  The algorithm leverages historical data patterns to make accurate predictions.
+ Exploratory Data Analysis (EDA):
  + Robust EDA is performed on historical flight data.
  + Insights gained from EDA inform the subsequent modeling steps.
 



![image](https://github.com/Gitamrit/Flight-Price-Prediction/assets/163405281/0daa1182-30a9-414f-9375-3c73a8f79331)


## Key Features of the Trained Machine Learning Model:

+ Diverse Model Training:
  + The model is trained using various machine learning algorithms.
  + By exploring different models, we enhance the chances of capturing complex patterns in the data.
+ Hyperparameter Optimization with Randomized Search CV:
  + Randomized Search Cross-Validation (CV) is employed to fine-tune hyperparameters.
  + This process systematically explores hyperparameter space, leading to better model performance.
+ Outlier Handling by Data Capping:
  + Outliers, which can skew predictions, are identified and capped.
  + Capping ensures that extreme values do not disproportionately influence the model.


## Technologies Used
Python, HTML/CSS,Requests, Render_template, Pandas, sci-kit learn, Pickel.

## How to use
Since the website is not hosted yet , you need to do it manually by hosting it locally . Clone the Github link, run python app.py on your terminal and open http://127.0.0.1:5000/ on your web browser.
