# Module-14-Project

# Jonathan Frazure

Machine Learning Trading Bot

In this Challenge, you’ll assume the role of a financial advisor at one of the top five financial advisory firms in the world. Your firm constantly competes with the other major firms to manage and automatically trade assets in a highly dynamic environment. In recent years, your firm has heavily profited by using computer algorithms that can buy and sell faster than human traders.

The speed of these transactions gave your firm a competitive advantage early on. But, people still need to specifically program these systems, which limits their ability to adapt to new data. You’re thus planning to improve the existing algorithmic trading systems and maintain the firm’s competitive advantage in the market. To do so, you’ll enhance the existing trading signals with machine learning algorithms that can adapt to new data.

Instructions:

Use the starter code file to complete the steps that the instructions outline. The steps for this Challenge are divided into the following sections:

Establish a Baseline Performance

Tune the Baseline Trading Algorithm

Evaluate a New Machine Learning Classifier

Create an Evaluation Report

Establish a Baseline Performance

In this section, you’ll run the provided starter code to establish a baseline performance for the trading algorithm. To do so, complete the following steps.

Open the Jupyter notebook. Restart the kernel, run the provided cells that correspond with the first three steps, and then proceed to step four.

Import the OHLCV dataset into a Pandas DataFrame.

Generate trading signals using short- and long-window SMA values.

Split the data into training and testing datasets.

Use the SVC classifier model from SKLearn's support vector machine (SVM) learning method to fit the training data and make predictions based on the testing data. Review the predictions.

Review the classification report associated with the SVC model predictions.

Create a predictions DataFrame that contains columns for “Predicted” values, “Actual Returns”, and “Strategy Returns”.

Create a cumulative return plot that shows the actual returns vs. the strategy returns. Save a PNG image of this plot. This will serve as a baseline against which to compare the effects of tuning the trading algorithm.

Write your conclusions about the performance of the baseline trading algorithm in the README.md file that’s associated with your GitHub repository. Support your findings by using the PNG image that you saved in the previous step.

Tune the Baseline Trading Algorithm

In this section, you’ll tune, or adjust, the model’s input features to find the parameters that result in the best trading outcomes. (You’ll choose the best by comparing the cumulative products of the strategy returns.) To do so, complete the following steps:

Tune the training algorithm by adjusting the size of the training dataset. To do so, slice your data into different periods. Rerun the notebook with the updated parameters, and record the results in your README.md file. Answer the following question: What impact resulted from increasing or decreasing the training window?
Hint To adjust the size of the training dataset, you can use a different DateOffset value—for example, six months. Be aware that changing the size of the training dataset also affects the size of the testing dataset.

Tune the trading algorithm by adjusting the SMA input features. Adjust one or both of the windows for the algorithm. Rerun the notebook with the updated parameters, and record the results in your README.md file. Answer the following question: What impact resulted from increasing or decreasing either or both of the SMA windows?

Choose the set of parameters that best improved the trading algorithm returns. Save a PNG image of the cumulative product of the actual returns vs. the strategy returns, and document your conclusion in your README.md file.

Evaluate a New Machine Learning Classifier

In this section, you’ll use the original parameters that the starter code provided. But, you’ll apply them to the performance of a second machine learning model. To do so, complete the following steps:

Import a new classifier, such as AdaBoost, DecisionTreeClassifier, or LogisticRegression. (For the full list of classifiers, refer to the Supervised learning page in the scikit-learn documentation.)

Using the original training data as the baseline model, fit another model with the new classifier.

Backtest the new model to evaluate its performance. Save a PNG image of the cumulative product of the actual returns vs. the strategy returns for this updated trading algorithm, and write your conclusions in your README.md file. Answer the following questions: Did this new model perform better or worse than the provided baseline model? Did this new model perform better or worse than your tuned trading algorithm?

Create an Evaluation Report

In the previous sections, you updated your README.md file with your conclusions. To accomplish this section, you need to add a summary evaluation report at the end of the README.md file. For this report, express your final conclusions and analysis. Support your findings by using the PNG images that you created.

# Imports

# Ignore warnings

In this section, you’ll run the provided starter code to establish a baseline performance for the trading algorithm. To do so, complete the following steps.

Open the Jupyter notebook. Restart the kernel, run the provided cells that correspond with the first three steps, and then proceed to step four.

Step 1: Import the OHLCV dataset into a Pandas DataFrame.

# Import the OHLCV dataset into a Pandas Dataframe

# Review the DataFrame

# Filter the date index and close columns

# Use the pct_change function to generate returns from close prices

# Drop all NaN values from the DataFrame

# Review the DataFrame

Step 2: Generate trading signals using short- and long-window SMA values.

# Set the short window and long window

# Generate the fast and slow simple moving averages (4 and 100 days, respectively)

# Review the DataFrame

# Initialize the new Signal column

# When Actual Returns are greater than or equal to 0, generate signal to buy stock long

# When Actual Returns are less than 0, generate signal to sell stock short

# Review the DataFrame

# Calculate the strategy returns and add them to the signals_df DataFrame

# Review the DataFrame

# Plot Strategy Returns to examine performance

# Assign a copy of the sma_fast and sma_slow columns to a features DataFrame called X

# Review the DataFrame

# Create the target set selecting the Signal column and assiging it to y

# Review the value counts

# Select the start of the training period

# Display the training begin date

# Select the ending period for the training data with an offset of 3 months

# Display the training end date

# Generate the X_train and y_train DataFrames

# Review the X_train DataFrame

# Generate the X_test and y_test DataFrames

# Review the X_test DataFrame

# Scale the features DataFrames

# Create a StandardScaler instance

# Apply the scaler model to fit the X-train data

# Transform the X_train and X_test DataFrames using the X_scaler

# From SVM, instantiate SVC classifier model instance
 
# Fit the model to the data using the training data
 
# Use the testing data to make the model predictions

# Review the model's predicted values

# Use a classification report to evaluate the model using the predictions and testing data

# Print the classification report

 precision    recall  f1-score   support

        -1.0       0.43      0.04      0.07      1804
         1.0       0.56      0.96      0.71      2288

    accuracy                           0.55      4092
   macro avg       0.49      0.50      0.39      4092
weighted avg       0.50      0.55      0.43      4092

# Create a new empty predictions DataFrame:

# Create a predictions DataFrame

# Add the SVM model predictions to the DataFrame

# Add the actual returns to the DataFrame

# Add the strategy returns to the DataFrame

# Review the DataFrame

# Plot the actual returns versus the strategy returns

Tune the Baseline Trading Algorithm
In this section, you’ll tune, or adjust, the model’s input features to find the parameters that result in the best trading outcomes. You’ll choose the best by comparing the cumulative products of the strategy returns.

Step 1: Tune the training algorithm by adjusting the size of the training dataset.

To do so, slice your data into different periods. Rerun the notebook with the updated parameters, and record the results in your README.md file.

Answer the following question: What impact resulted from increasing or decreasing the training window?

Step 2: Tune the trading algorithm by adjusting the SMA input features.

Adjust one or both of the windows for the algorithm. Rerun the notebook with the updated parameters, and record the results in your README.md file.

Answer the following question: What impact resulted from increasing or decreasing either or both of the SMA windows?

Step 3: Choose the set of parameters that best improved the trading algorithm returns.

Save a PNG image of the cumulative product of the actual returns vs. the strategy returns, and document your conclusion in your README.md file.

Evaluate a New Machine Learning Classifier

In this section, you’ll use the original parameters that the starter code provided. But, you’ll apply them to the performance of a second machine learning model.

Step 1: Import a new classifier, such as AdaBoost, DecisionTreeClassifier, or LogisticRegression. (For the full list of classifiers, refer to the Supervised learning page in the scikit-learn documentation.)

# Import a new classifier from SKLearn

# Initiate the model instance

# Fit the model using the training data

# Use the testing dataset to generate the predictions for the new model

# Review the model's predicted values

Step 3: Backtest the new model to evaluate its performance.

Save a PNG image of the cumulative product of the actual returns vs. the strategy returns for this updated trading algorithm, and write your conclusions in your README.md file.

Answer the following questions: Did this new model perform better or worse than the provided baseline model? Did this new model perform better or worse than your tuned trading algorithm?

# Use a classification report to evaluate the model using the predictions and testing data

# Print the classification report
print(signal_predictions)
              precision    recall  f1-score   support

        -1.0       0.44      0.33      0.38      1804
         1.0       0.56      0.66      0.61      2288

    accuracy                           0.52      4092
   macro avg       0.50      0.50      0.49      4092
weighted avg       0.51      0.52      0.51      4092

# Create a new empty predictions DataFrame:

# Create a predictions DataFrame

# Add the SVM model predictions to the DataFrame

# Add the actual returns to the DataFrame

# Add the strategy returns to the DataFrame

# Review the DataFrame

# Plot the actual returns versus the strategy returns


##  Conclusion  ##

I noticed in each situation that extending the training window increased performance every so slightly.  There is not a real prohibitive difference in return profile overall between the two methodologies, but I tend to lean toward the SVC model as the correct one to use to tweak parameters and gain meaningful improvement with some work.

