# Vegan Youtube

### Predicting Youtube engagement for vegan and animal rights based content

This supervised learning project explores what makes vegan and animal rights based Youtube content successful in terms of engagement. Using the official Youtube API, machine learning models, and model interpretability tools, I investigate which features are associated with view rate (view count / number of days published) and what their relationship is.

The population of videos being analysed is videos with an animal rights based message and or made with the intent to encourage others to live vegan (the ethical principal that animals are entitled to rights). This can include cooking, lifestyle and health videos made with that intent. It can also include videos from non-vegan channels such as speeches, interviews and debates, so long as there is someone in these videos arguing for the ethical principal of veganism. Excluded are diet-challenge style videos focused on experimentation and personal experience, rather than advocacy.

The goal of this project is to gain insights into what features are associated with higher view rates, so as to make recommendations to vegan content creators to optimize their reach.

## Objectives

- Scrape Youtube metadata using official API
- Prepare data for modeling
- Model to predict view rate using RMSE and MedAE as metrics
- Interpret model behaviour using SHAP and Partial Dependancy Plots
- Identify actionable insights for content creators

## Tools & Technlogies

- **Language:** Python
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, xgboost, shap, googleapiclient, pickle, datetime, zoneinfo, os, re, ast, collections, itertools
- **API:** Youtube Data API v3
- **IDE:** Jupiter notebook (VS code)

## Modeling Aproach

- **Target variable:** Box cox transformed view rate (view count / number of days published)
- **Models used:** Linear regression (with lasso for feature selection), decision tree, random forest & xgboost
- **Loss function:** MSE
- **Evaluation metrics:** RMSE, MedAE & Residual analysis (in back transformed view rate units)

## Summary Results

- **Linear regression:** RMSE: 484.06 (95% CI=362.92, 598.35), MedAE: 18.52 (95% CI=14.5, 23.97)
- **Decision tree:** RMSE: 548.46 (95% CI=352.47, 717.42), MedAE: 16.67 (95% CI=11.72, 24.9)
- **Random forest:** RMSE: 518.92 (95% CI=326.41, 684.91), MedAE: 13.66 (95% CI=10.44, 15.51)
- **XGBoost:** RMSE: 519.54 (95% CI=329.84, 683.8), MedAE: 12.22 (95% CI=9.3, 15.4)

## Insights & Recommendations

Of the features that were used to train the model, not all are actionable in terms of what content creators can do to influence. For example while subscriber count consistently showed to be the most impactful variable, this is not something a creator can increase directly. For this reason features were split into actionable and non-actionable, and only the actionable were investigated.

The analysis using two independent models (Random Forest & XGBoost) found terminology used in the title and description of a video to be the most impactful actionable variable to consistently impact the prediction of view rate. In particular, use of the term "outreach" is associated with lower view rate, especially when used with language mentioning animals and veganism. This could be that it is seen to be associated with charity, fundrasing or other topics that get lower click-through rate. Instead creators could try alternative wording such as "public engagement" or just using titles ommiting the word entirely. The avoidence of the term had a mean effect size on view rate of 5.5 views per day. Creators can also try A/B testing titles with and without the word to see what works best. The word "animal" itself is also consistently correlated with lower predictions across both models, this could be because in vegan content the term is overused or attracts a niche audience. It could be more effective to mention animals by specific species and use emotional language e.g. "baby piglets". The avoidence of this term had a mean effect size of 3.8.

As for what to include, the term "vegan" consistently contributed to positive signal when included in titles and descriptions, this could be a strong identity keyword that signals to Youtube to recommend for that audience. So long as it's relevant to the content it is recommended to explicitly use the term "vegan" in the title and or description. Its inclusion had a mean effect size of only 1.4, however the models tended to predict higher view rates when it was associated with animal rights based terminology. Despite "animal", and "rights" individually having negative impact, when used together and with vegan terminology, it is recommended to use this language. The term "instagram" also showed strong positive impact on predictions across models (mean effect size 3.5), this is from creators linking their accounts in the video description. This might correlate with creators who have cross-platform promotion and a loyal following. There is also cross-model consensus that using the term "vegan" in combination with linking your Instagram and requesting support from your audience is associated with higher view rate. If you have an Instagram account, it is recommended to always link it in the video description, especially if you have a large following.

## How to Run

```bash
git clone https://github.com/gh4rris/vegan_youtube.git
cd vegan_youtube
pip install -r requirements.txt
```

An API_KEY enviroment variable created using a Google account is required to run [01_scraping.ipynb](/01_scraping.ipynb), however this isn't necessary as the data has already been scrapped and stored in the [/data](/data/) directory. The remainder of the notebooks can be run in order:

- [02_wrangling_eda.ipynb](/02_wrangling_eda.ipynb)
- [03_modeling_linreg.ipynb](/03_modeling_linreg.ipynb)
- [04_modeling_dt_rf.ipynb](/04_modeling_dt_rf.ipynb)
- [05_interpretation](/05_interpretation.ipynb)

## Data Collection [01_scraping.ipynb](/01_scraping.ipynb)

The Youtube client object was built using the official Google python client library found [here](https://github.com/googleapis/google-api-python-client). An API key was then generated using a google account, and set as an environment variable. A video search was then done for key search terms such as "animal liberation" and "vegan speech", also for key animal rights content creators such as "joey carbstrong" and "earthling ed". Each search returned the video IDs for the first 100 results. The list of IDs was de-duplicated, and data on each individual video was collected, including the channel ID. As category returns an ID rather than the name, an additional search was used to collect the names. Finally the channel ID collected earlier was used to collect additional channel info for each video, such as subscribber and video count. All data was saved into the [/data](/data/) directory as [videos_unprepared.csv](/data/videos_unprepared.csv).

## Feature Engineering & Exploratory Data Analysis [02_wrangling_eda.ipynb](/02_wrangling_eda.ipynb)

I first extracted the hour, weekday and month of the year the video was published from the date, as well as how many days the video had been published for. Next, as the category and channel data were in seperate csv files, I imported them and merged everything into one dataframe. The duration was initially in a string format e.g. "PT1H32M15S" for 1 hour, 32 mins and 15 seconds. I converted this into total seconds for a consistent numarical value, then videos under 180 seconds (3mins) were assigned an _is_short_ binary variable for Youtube shorts. _definition_ and _caption_ were also converted to binary, for definition 0 = sd & 1 = hd, and for caption 0 = False, 1 = True. I filtered out any channel names that had videos in the data that weren't representative of the target population e.g. they were talking about veganism negatively or not at all, then further filtered out videos under 7 days old and unwanted video subjects from the remaining channels. The target variable _view_rate_ was then calculated by dividing _view_count_ by _published_duration_days_.

For tags, as there were almost 5000 unique ones, I just created features for the top 11 and also a seperate one for total tag count.

![Top tags](/images/tag_frequency.png)

Next I created features for title and decription length, and also used a tf-idf vectorizer to vectorize the term importance in both, ignoring small words like "a" and "the". This created an extra 300 features of terms.

Checking numerical variable distributions, it can be seen that there is a strong right skew where the majority of variables like view and subscribber counts are at the lower end, and only a small minority go viral:

![Variable distributions](/images/variable_distributions.png)

Then applying a log transformation to these variables, a cube root transformation to the _published_duration_days_ and _description_length_ as its skew is less extreme, and a box cox transformation to _view_rate_ as it's skew is the most extreme. This is to normalize the data so that variables with larger scales don't dominate the results:

![Transformed distributions](/images/transformed_variable_distributions.png)

For categories, I sorted the value counts and then for any that had less than 10 videos I combined into an "Other" category.

The _box_view_rate_ distributions for the top 5 categories:

![Category distributions](/images/category_distributions.png)

Science & Technology videos have the highest median view rate, as well as the least amount of variance suggesting that these types of videos are safe in terms of performing well, but don't have the same potential to go viral as People & Blogs or Nonprofits & Activism. However, these two categories have high variability, and carry the risk of performing much worse. Entertainment and Science & Technology both show some outliers, investigating these shows videos that are all representative of the population but went viral or underperformed, so they were kept in. The categories were then one-hot encoded, which gave me a feature column for every category with 0 for False and 1 for True.

The numerical features were then formatted into a heatmap to compare correlations:

![Numerical heatmap](/images/numerical_heatmap.png)

Unsuprisingly the view rates strongest positive correlations are with channel subscriber and channel view count. It also has a somewhat negative correlation with the number of days the video has been published for, which makes sense as the rate of views will eventually decline over time.

The prepared data was finally saved into the [/data](/data/) directory as [videos_prepared.csv](/data/videos_prepared.csv).

## Model Training & Evaluation [03_modeling_linreg.ipynb](/03_modeling_linreg.ipynb), [04_modeling_dt_rf.ipynb](/04_modeling_dt_rf.ipynb)

The target variable I am modelling to predict is the box cox transformed view rate _box_view_rate_. This means the predictions will need to be scaled back to to get in number of views. The first approach to this was to try and find a linear relationship between the features and the target. I standardized the data, as some of the features have different ranges, and then applied a lasso regression for feature selection, to find what the most relevant features were for predicting view rate. I tried multiple alpha values and looped through them to get the smallest RMSE. Using the optimal alpha value of 0.01, the features were reduced from 337 to 115. A linear model was fitted to the training data for just these features, and then applying this model to the test data the RMSE was 484.06 (once back transformed). Although the RMSE is quite high, the median absolute error (MedAE) is only 18.52 suggesting the model is accurate most of the time, but there are some large misses that skew the residuals.

The residuals for the predictions were then plotted. Looking at this, we can see that the bulk of the models predictions were below 300 view rate, and that these tended to be reletively accurate, however there were some outliers it tended to underpredict. When the model made predictions between 400 and 1000, it would overpredict. There is one extreme outlier that the model underpredicted by over 4000, when predicted 1600.

![Linreg residuals](/images/linreg_residual_plot.png)

This pattern of change in residual variance at different prediction levels suggests a more flexible model may be appropriate, so next I tried a decision tree. For this I used hyperparameter tuning to find the optimal parameters, and fitted a model with these to the training data. The RMSE for the decision tree on the test data was 548.46. A bit of a downgrade, so I tried to improve using a random forest. This time for hyperparameter tuning I used RandomSearchCV to search a wide range of randomly selected parameters, and then using the best parameter results, used GridSearchCV to fine-tune a narrower range. Once I fit the random forest with the optimal parameters and scored it against the test data, the RMSE was 518.92 (95% CI=326.41, 684.91). This is still larger than the linear regression, however the MedAE has now decreased to 13.66 (95% CI=10.44, 15.51). This suggests an improved accuracy for smaller view rates, although large misses have grown, skewing the residuals further.

This is confirmed looking at the residual plot. It is one residual in particular over 5000 heavily skewing the distribution causing the increased RMSE.

![RandForest residuals](/images/rf_residual_plot.png)

The same hyperparameter tuning method was applied to xgboost acheiving an RMSE of 519.54 (95% CI=329.84, 683.8) and a MedAE of 12.22 (95% CI=9.3, 15.4). Roughly the same results as the random forest.

The residual plot for the xgboost is similar to the random forest. Where the random forest made a few more predictions over 800, the xgboost was more conservative:

![XGBoost residuals](/images/xgb_residual_plot.png)

## Interpretation [05_interpretation](/05_interpretation.ipynb)

The random forest and the xgboost were the two best performing models, with roughly equivalent evaluation metrics, but slightly varying residuals suggesting they are picking up on some different patterns. For this reason, they where both chosen to examine for patterns correlated with view rate, and only where there is cross-model consensus were recommendations made.

First the mean absolute shap values for each feature were calculated and sorted in decending order. This gave a list of the most impactful features on view rate. All non-actionable features were filtered out, as the goal is to find effective features content creators can act upon. An effect size was then calculated using the difference in partial dependence range. This gave a figure in mean views per day that the model predicted extra from the worst setting of the given feature, to the best setting (all else held constant).

Since this process was done for both the random forest and xgboost individually, the results were then combined to calculate a mean effect size of features both models agreed were important. Any features were the models disagreed on the direction of the imporovement were filtered out, which left 11 top actionable terms.

- [Random Forest PDP for Top Actionable Features](/images/rf_pdp_actionable.png)
- [XGBoost PDP for Top Actionable Features](/images/xgb_pdp_actionable.png)

![Random forest beeswarm](/images/rf_beeswarm_actionable.png)

![XGBoost beeswarm](/images/xgb_beeswarm_actionable.png)

All of the top actionable features are terms from the tf-idf vectorizer, so I wanted to examine any interaction affects these top terms have with other terms they are related to. To do this, first I set thresholds at which the term was considered high. If it was a term positively correlated with view rate, then the threshold was set at the mean of the best settings for both models. If it was a term negatively associated with view rate, then the threshold was set at 0.

Using these thresholds all video text was collected for each of the top features where the term usage was above its threshold and then the top 20 associated terms were added to a list. Cycling through each top feature, associated term combination, I compared if the median prediction increased or decreased when both terms were high, or when the top feature was high but associated term wasn't. If both models agreed on the direction, I took the mean, if not, I discarded it. Also it is worth noting that some words were skipped, as they weren't included as features by the tf-idf vectorizer, especially small common words like "a" and "the".

This gave me pairs of associated words, but I also wanted to try sets of 3. So, I did the same again iterating each top feature with each combination of 2 other associated terms. Then compared the median of all terms being high to the top feature being high and one of the associated terms not. Again if models agreed I took the mean, if not I discarded it.
