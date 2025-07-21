# Vegan Youtube

## Predicting youtube engagement for vegan and animal rights based content

This supervised learning project explores what makes vegan and animal rights based youtube content successful in terms of engagement, using view count as a target metric. Using the official Youtube API, machine learning models, and model interpretability tools, I investigate which features are associated with view count and what their relationship is.

## Objectives

- Scrape Youtube metadata using official API
- Prepare data for modeling
- Model to predict view count using R2 and RMSE as metrics
- Interpret model behaviour using SHAP and Partial Dependancy Plots
- Identify actionable insights for content creators

## Tools & technlogies

- **Language:** Python
- **Libraries:** Pandas, Numpy, Matplotlib, Seaborn, scikit-learn, shap, googleapiclient, os, re, ast
- **API:** Youtbe Data API v3
- **IDE:** Jupiter notebook (VS code)

## Data collection

In the first file _01_scraping.ipynb_ the Youtube client object was built using the official Google python client library found [here](https://github.com/googleapis/google-api-python-client). An API key was then generated using a google account, and set as an environment variable. A video search was then done for key search terms such as "animal liberation" and "vegan speech", also for key animal rights content creators such as "joey carbstrong" and "earthling ed". Each search returned the video IDs for the first 100 results. The list of IDs was de-duplicated, and data on each individual video was collected, including the channel ID. As category returns an ID rather than the name, an additional search was used to collect the names. Finally the channel ID collected earlier was used to collect additional channel info for each video, such as subscribber and video count. All data was saved into the _/data_ directory as _videos_unprepared.csv_.

## Feature engineering

In _02_cleaning.ipynb_ I first extracted the hour, weekday and month of the year the video was published from the date. Next, as the category and channel data were in seperate csv files, I imported them and merged everything into one dataframe. The duration was initially in a string format e.g. "PT1H32M15S" for 1 hour, 32 mins and 15 seconds. I converted this into total seconds for a consistent numarical value. Binary variables Were turned into 1s and 0s, for definition 0 = sd & 1 = hd, and for caption 0 = False, 1 = True. I removed videos with the hastag #shorts in the title and also filtered out any channel names that had videos in the data that weren't representative of the target population e.g. they were talking about veganism but not promoting animal rights. Next I one-hot encoded the categories, dropping the dedundant first column which was "Autos & Vehicles". This gave me a feature column for every category with 0 for False and 1 for True.

For tags, as there were almost 5000 unique ones, I just created features for the top 11 and also a seperate one for total tag count.

![Top tags](/images/tag_frequency.png)

Next I created features for title and decription length, and also used a tf-idf vectorizer to vectorize the term importance in both, ignoring small words like "a" and "the". This created an extra 300 features of terms.

Checking numerical variable distributions, it can be seen that there is a strong right skew where the majority of variables like view and subscribber counts are at the lower end, and only a small minority go viral:

![Variable distributions](/images/variable_distributions.png)

Then applying a log transformation to these variables, and a cube root transformation to the description length as its skew is less extreeme. This is to normalize the data so that variables with larger scales don't dominate the results:

![Transformed distributions](/images/transformed_variable_distributions.png)

The prepared data was finally saved into the _/data_ directory as _videos_prepared.csv_.

## Modeling approach

The target variable I am modelling to predict is the log transformed view count _log_view_count_. This means the predictions will need to be scaled back to to get in number of views. The first approach to this was to try and find a linear relationship between the features and the target. I standardized the data, as as some of the features have different ranges, and then applied a lasso regression for feature selection, to find what the most relevant features were for predicting view count. I tried multiple alpha values and looped through them to get the highest R2 and lowest root mean squared error (RMSE). Using the optimal alpha value of 0.1, the features were reduced from 337 to 25. A linear model was fitted to the training data for just these features and then on the test data using these features the R2 was about 0.53 and RMSE about 1.81 (log).

Looking at the models top coefficients, subscribber count has the strongest association in explaining the variance in view count, followed by the channels total view count and thenwhether or not is of the Music category. All of the top features appear to have a positive correlation with the target, with the exception of the term "surge" which is negative:

![Linreg features](/images/linreg_top_features.png)

The residuals for the predictions were then plotted. Looking at this, we can see that at the low prediction levels up to log 6, the model tended to over predict, amost by as much as log 8. Then at the mid prediction levels of about log 6 to 12 the residuals seem evenly spread out, but then above log 12 all residuals appear to be negative and so the model is overpredicting, although not as extreeme as the underpredictions at the lower end.

![Linreg residuals](/images/linreg_residual_plot.png)

This pattern in residuals at varying levels of prediction suggests a more flexible model may be appropriate, so next I tried a decision tree. For this I used hyperparameter tuning to find the optimal parameters, and fitted a model with these to the training data. The R2 for the decision tree on the test data was about 0.63, and RMSE was about 1.59. An improvement, but tried to see if I could do better using a random forest. This time for hyperparameter tuning I used RandomSearchCV to search a wide range of randomly selected parameters, and then using the best parameter results, used GridSearchCV to fine-tune a narrower range. Once I fit the random forest with the optimal parameters and scored it against the test data, the R2 was about 0.73 and RMSE about 1.38.

There doesn't appear to be any clear pattern for the residuals of the random forest, indicating the model is well-fitted.

![RandForest residuals](/images/rf_residual_plot.png)
