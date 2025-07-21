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

In the first file '01_scraping.ipynb' the youtube client was built using the official google python client library found [here](https://github.com/googleapis/google-api-python-client). An API key was then generated using my google account, and set as an environment variable. A video search was then done for key search terms such as "animal liberation" and "vegan speech", also for key animal rights content creators such as "joey carbstrong" and "earthling ed". Each search returned the video IDs for the first 100 results. The list of IDs was de-duplicated, and data on each individual video was collected, including the channel ID. As category returns an ID rather than the name, an additional search was used to collect the names. Finally the channel ID collected earlier was used to collect additional channel info for each video, such as subscribber and video count. All data was saved into the '/data' directory as 'videos_nprepared.csv'.

## Feature engineering

In '02_cleaning.ipynb' I first extracted the hour, weekday and month of the year the video was published from the date. Next, as the category and channel data were in seperate csv files, I imported them and merged everything into one dataframe. The duration was initially in a string format e.g. "PT1H32M15S" for 1 hour, 32 mins and 15 seconds. I converted this into total seconds for a consistent numarical value. Binary variables I turned into 1s and 0s, for definition 0 = sd & 1 = hd, ad for caption 0 = False, 1 = True. I removed videos with the hastag #shorts in the title and also filtered out any channel names that had videos in the data that weren't representative of the target population e.g. they were talking about veganism but not promoting animal rights. Next I one-hot encoded the categories, dropping the dedundant first column which was "Autos & Vehicles". This gave me a feature column for every category with 0 for False and 1 for True.

For tags, as there were almost 5000 unique ones, I just created features for the top 11 and also a seperate one for total tag count.

![Top tags](/images/tag_frequency.png)

Next I created features for title and decription length, and also used a tf-idf vectorizer to vectorize the term importance in both, ignoring small words like "a" and "the". This created an extra 300 features of terms.

Checking numerical variable distributions, it can be seen that there is a strong right skew where the majority of view and subscribber counts are at the lower end, and only a small minority go viral:

![Variable distributions](/images/variable_distributions.png)

Then applying a log transformation to these variables, and a cube root transformation to the description length as it is less extreeme. This is to normalize the data so that variables with larger scales don't dominate the results:

![Transformed distributions](/images/transformed_variable_distributions.png)

The prepared data was finally saved into the '/data' directory as 'videos_prepared.csv'.
