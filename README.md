# Vegan Youtube

### Predicting Youtube engagement for vegan and animal rights based content

This supervised learning project explores what makes vegan and animal rights based Youtube content successful in terms of engagement. Using the official Youtube API, machine learning models, and model interpretability tools, I investigate which features are associated with view rate (view count / number of days published) and what their relationship is.

The population of videos being analysed is videos with an animal rights based message and or made with the intent to encourage others to live vegan (the ethical principal that animals are entitled to rights). This can include cooking, lifestyle and health videos made with that intent. It can also include videos from non-vegan channels such as speeches, interviews, debates, so long as there is someone in these videos arguing for the ethical principal of veganism. It would not however include videos of someone attempting a vegan diet for a period of time, even if the experience is spoken about positively and they encourage others to try it.

The goal of this project is to gain insights into what features are associated with higher view rates, so as to make recomendations to vegan content creators to optimize their reach.

## Objectives

- Scrape Youtube metadata using official API
- Prepare data for modeling
- Model to predict view rate using R2 and RMSE as metrics
- Interpret model behaviour using SHAP and Partial Dependancy Plots
- Identify actionable insights for content creators

## Tools & Technlogies

- **Language:** Python
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, xgboost, shap, googleapiclient, pickle, datetime, zoneinfo, os, re, ast
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

The analysis using two independent models (Random Forest & XGBoost) found certain terminology used in the title and description of a video to cosistently impact the prediction of view rate. In particular, use of the term "outreach" is associated with lower view rate. This could be because it's seen to be associated with charity, fundrasing or other topics that get lower click-through rate. Instead creators could try alternative wording such as "public engagement" or just using titles ommiting the word entirely. Creators can also try A/B testing titles with and without the word to see what works best. Another word consistently correlated with lower predictions across both models is "animal", this could be because in vegan content the term is overused or attracts a niche audience. It could be more effective to mention animals by specific species and use emotional language e.g. "baby piglets".

As for what to include, the term "vegan" consistently contributed to positive signal when included in titles and descriptions, this could be a strong identity keyword that signals to Youtube to recommend for that audience. So long as it's relevant to the content it is recommended to explicily use the term "vegan" in the title and or description. The term "instagram" also showed strong positive impact on predictions across models, this is from creators linking their accounts in the video description. This might correlate with creators who have cross-platform promotion and a loyal following. If you have an instagram account, it is recommended to always link it in the video description, especially if you have a large following.

## Data Collection [01_scraping.ipynb](/01_scraping.ipynb)

The Youtube client object was built using the official Google python client library found [here](https://github.com/googleapis/google-api-python-client). An API key was then generated using a google account, and set as an environment variable. A video search was then done for key search terms such as "animal liberation" and "vegan speech", also for key animal rights content creators such as "joey carbstrong" and "earthling ed". Each search returned the video IDs for the first 100 results. The list of IDs was de-duplicated, and data on each individual video was collected, including the channel ID. As category returns an ID rather than the name, an additional search was used to collect the names. Finally the channel ID collected earlier was used to collect additional channel info for each video, such as subscribber and video count. All data was saved into the [/data](/data/) directory as [videos_unprepared.csv](/data/videos_unprepared.csv).

## Feature Engineering & Exploratory Data Analysis [02_wrangling_eda.ipynb](/02_wrangling_eda.ipynb)

I first extracted the hour, weekday and month of the year the video was published from the date, as well as how many days the video had been published for. Next, as the category and channel data were in seperate csv files, I imported them and merged everything into one dataframe. The duration was initially in a string format e.g. "PT1H32M15S" for 1 hour, 32 mins and 15 seconds. I converted this into total seconds for a consistent numarical value, then videos under 180 seconds (3mins) were assigned an _is_short_ binary variable for Youtube shorts. _definition_ and _caption_ were also converted to binary, for definition 0 = sd & 1 = hd, and for caption 0 = False, 1 = True. I filtered out any channel names that had videos in the data that weren't representative of the target population e.g. they were talking about veganism negatively or not at all, then further filtered out unwanted subjects from the remaining channels. The target variable _view_rate_ was then calculated by dividing _view_count_ by _published_duration_days_.

For tags, as there were almost 5000 unique ones, I just created features for the top 11 and also a seperate one for total tag count.

![Top tags](/images/tag_frequency.png)

Next I created features for title and decription length, and also used a tf-idf vectorizer to vectorize the term importance in both, ignoring small words like "a" and "the". This created an extra 300 features of terms.

Checking numerical variable distributions, it can be seen that there is a strong right skew where the majority of variables like view and subscribber counts are at the lower end, and only a small minority go viral:

![Variable distributions](/images/variable_distributions.png)

Then applying a log transformation to these variables, a cube root transformation to the _published_duration_days_ and _description_length_ as its skew is less extreme, and a box cox transformation to _view_rate_ as it's skew is the most extreme. This is to normalize the data so that variables with larger scales don't dominate the results:

![Transformed distributions](/images/transformed_variable_distributions.png)

Then looked at the _view_rate_ distributions for the top 5 categories:

![Category distributions](/images/category_distributions.png)

Science & Technology videos have the highest median view rate, as well as the least amount of variance suggesting that these types of videos are safe in terms of performing well, but don't have the same potential to go viral as People & Blogs or Nonprofits & Activism. However, these two categories have high variability, and carry the risk of performing much worse. Entertainment and Science & Technology both show some outliers, investigating these shows videos that are all representative of the population but went viral or overperformed, so they were kept in. The categories were then one-hot encoded, which gave me a feature column for every category with 0 for False and 1 for True.

The numerical features were then formatted into a heatmap to compare correlations:

![Numerical heatmap](/images/numerical_heatmap.png)

Unsuprisingly the view rates strongest positive correlations are with channel subscriber and channel view count. It also has a somewhat negative correlation with the number of days the video has been published for, which makes sense as the rate of views will eventually decline over time.

The prepared data was finally saved into the [/data](/data/) directory as [videos_prepared.csv](/data/videos_prepared.csv).

## Model Training & Evaluation [03_modeling_linreg.ipynb](/03_modeling_linreg.ipynb), [04_modeling_dt_rf.ipynb](/04_modeling_dt_rf.ipynb)

The target variable I am modelling to predict is the box cox transformed view rate _box_view_rate_. This means the predictions will need to be scaled back to to get in number of views. The first approach to this was to try and find a linear relationship between the features and the target. I standardized the data, as some of the features have different ranges, and then applied a lasso regression for feature selection, to find what the most relevant features were for predicting view rate. I tried multiple alpha values and looped through them to get the smalles RMSE. Using the optimal alpha value of 0.01, the features were reduced from 337 to 115. A linear model was fitted to the training data for just these features, and then on the test data using these features the RMSE was 484.06 (once back transformed). Although the RMSE is quite high, the median absolute error (MedAE) is only 18.52 suggesting the model is accurate most of the time, but there are some large misses that skew the residuals.

The residuals for the predictions were then plotted. Looking at this, we can see that the bulk of the models predictions were below 300 view rate, and that these tended to be reletively accurate, however there were some outliers it tended to underpredict. When the model made predictions between 400 and 1000, it would overpredict. There is one extreme outlier that the model underpredicted by over 4000, when predicted 1600.

![Linreg residuals](/images/linreg_residual_plot.png)

This pattern of change in residual variance at different residual levels suggests a more flexible model may be appropriate, so next I tried a decision tree. For this I used hyperparameter tuning to find the optimal parameters, and fitted a model with these to the training data. The RMSE for the decision tree on the test data was 548.46. A bit of a downgrade, so I tried to improve using a random forest. This time for hyperparameter tuning I used RandomSearchCV to search a wide range of randomly selected parameters, and then using the best parameter results, used GridSearchCV to fine-tune a narrower range. Once I fit the random forest with the optimal parameters and scored it against the test data, the RMSE was 518.92 (95% CI=326.41, 684.91). This is still larger than the linear regression, however the MedAE has now decreased to 13.66 (95% CI=10.44, 15.51). This suggests an improved accuracy for smaller view rates, although large misses have grown, skewing the residuals further.

This is confirmed looking at the residual plot. It is one residual in particular over 5000 heavily skewing the distribution causing the increased RMSE.

![RandForest residuals](/images/rf_residual_plot.png)

The same hyperparameter tuning method was applied to xgboost acheiving an RMSE of 519.54 (95% CI=329.84, 683.8) and a MedAE of 12.22 (95% CI=9.3, 15.4). Roughly the same results as the random forest.

The residual plot for the xgboost is similar to the random forest. Where the random forest made a few more predictions over 800, the xgboost was more conservative:

![XGBoost residuals](/images/xgb_residual_plot.png)

## Interpretation [05_interpretation](/05_interpretation.ipynb)

### Linear regression

Looking at the models top coefficients, channel view count has the strongest association in explaining the variance in video view rate, followed by the subscriber count and then the term "vegangains". Most of the top features appear to have a positive correlation with the target, with the exception of the duration and the terms "davidractivism" and "join" which are negative:

![Linreg features](/images/linreg_top_features.png)

When sorted by highest channel view count or subscriber count with the largest view rates, we can see they share a lot of the same videos. Such as 5x "TEDx Talks", 2x "Brut India" videos and a "Big Think" video. This is likely due to the fact these two variables are highly correlated as we saw in the heat map. It is also worth noting that a lot of these videos are on non vegan related channels, but contain big names, whether known for veganism, or other things e.g. Ed Winters, Moby, Joaquin Phoenix, Joey Carbstrong and Peter Singer.

The third largest coefficient is the term "vegangains". Looking at the highest view rate videos that contain this term are a combination of fittness videos and debates. The noteable standout is the highest rated video by the channel "Turkey Tom", a documentary featuring and interviewing the youtuber.

### Random forest

The top features for the random forest were subscribers, days published & channel view count:

![Randforest features](/images/rf_top_feature_importance.png)

Using a partial dependence plot we can see that the average predicted view rate for a video with under 20 thousand subscribers is 2. There is an upward trend from this point and by 160 thousand subscribers predictions are averaging 78 per day with all other features held constant. The channel view count is relatively flat, as it is highly correlated with subscriber count and the model is relying on this more. As number of videos uploaded to a chnnel increase, there is a downward trend in view rate. Rates fall from about 30 at 20 videos to 15 at 1000 videos.

![RF pdp channel stats](/images/rf_pdp_channel_stats.png)

Differences in time of upload are very small, although there seems to be a pattern of more activity between 12 and 6pm. And at the bginning of the week (Monday & Tuesday).

![RF pdp time stats](/images/rf_pdp_time_stats.png)

Video duration, title length, description length, tag count and whether or not the video is a short all seem to have very little effect on the models prediction. However, how long the video has been published shows a clear downward trend in predictions. This makes sense, as videos that have been published for longer are likely to decline in popularity over time. Videos published for 60 days average at about 60 view rate, videos published for 500 days at about 20, and videos published for 2000 days at about 6.

![RF pdp misc variables](/images/rf_pdp_misc_variables.png)

The beeswarm shows the shap values of the most important terms used in the video titles and descriptions. Most notably the term "outreach" has a strong negative impact on view count. This could be because outreach videos are more of a niche activism focused genre, it could also be correlated with other features that are having a negative impact. The terms "instagram" and "gains" have a strong positive impact.

![RF term beeswarm](/images/rf_term_beeswarm.png)

While most categories in the dataset have little to no impact on view rate, Education had the strongest positive impact, followed by People & Blogs. Nonprofits & Activism and Pets & Animals both had negative impact:

![RF category beeswarm](/images/rf_category_beeswarm.png)

### XGBoost

The top features for the xgboost were subscribers, channel view count & the term "music":

![XGB features](/images/xgb_top_feature_importance.png)

The xgboost shows a similar positive trend in predictions for subscriber count as the random forest and negative trend for channel video count. It also seems to rely on channel view count for predictive power, where for the random forest most of that was absorbed by subscriber count:

![XGB pdp channel stats](/images/xgb_pdp_channel_stats.png)

Again differences in time of upload are small, but patterns that match with the random forest are that early and late in the day are the least performing times to upload and Monday & Tuesday are the best performing weekdays:

![XGB pdp time stats](/images/xgb_pdp_time_stats.png)

Published duration is showing the same downward pattern as it did in the random forest model and video duration, title length, description length, tag count and whether or not the video is a short all still have little affect on prediction:

![XGB pdp misc variables](/images/xgb_pdp_misc_variables.png)

The xgboost is still picking up on the same negative impact of the term "outreach" and positive impact of terms "instagram" and "gains". It is also picking up on a larger positive impact for the term "music":

![XGB term beeswarm](/images/xgb_term_beeswarm.png)

## Further investigation [05_investigation_shap.ipynb](/05_investigation_shap.ipynb)

### Outreach

Since containing the term "outreach" had a negative global impact in the beeswarm, I decided to look at the local impact of the highest viewed outreach videos using waterfall plots. Looking at these we can see they are all getting most of their negative impact from subscriber count, and sometimes channel view count. Channel video count is the main positive impact on views for these videos, which as we saw from the pdp earlier video count has a negative global influence, suggesting that channels that upload outreach videos generally have fewer videos on average.

From this we can see that although there is some small negative signal from containing the term "outreach", for the most part they aren't being inherently penalized, but rather are coming from smaller, less established channels with less subscribers and channel views. Possible recommendations: A/B test alternative titles that omit the word "outreach" entirely to see if there's any effect on view count, or mix outreach with broader categories, or do collabs to build subscribers.

- [Vegan Outreach 485](/images/local_shap/vegan_outreach_485.png)
- [James Kite 671](/images/local_shap/james_kite_671.png)
- [James Kite 464](/images/local_shap/james_kite_464.png)
- [Clif Grant 380](/images/local_shap/clif_grant_380.png)
- [The Cranky Vegan 683](/images/local_shap/the_cranky_vegan_683.png)

### Music

There were only two videos in the test data that were of the Music category, both with low view counts, and both of which were overpredicted by the model. Looking at the waterfall plots we can see that while being in the Music category does have some positive signal, it is outweighed by the low subscriber count. This is probably why the regression line for Music is flat, because while it does have a positive impact, less established channels with fewer subscribers will still get fewer views.

- [The Green Note 647](/images/local_shap/the_green_note_647.png)
- [禅 207](/images/local_shap/禅_207.png)
