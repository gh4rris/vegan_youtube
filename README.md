# Vegan Youtube

### Predicting Youtube engagement for vegan and animal rights based content

This supervised learning project explores what makes vegan and animal rights based Youtube content successful in terms of engagement. Using the official Youtube API, machine learning models, and model interpretability tools, I investigate which features are associated with view count and what their relationship is.

The population of videos being analysed is videos with an animal rights based message and or made with the intent to encourage others to live vegan (the ethical principal that animals are entitled to rights). This can include cooking, lifestyle and health videos made with that intent. It can also include videos from non-vegan channels such as speeches, interviews, debates, so long as there is someone in these videos arguing for the ethical principal of veganism. It would not however include videos of someone attempting a vegan diet for a period of time, even if the experience is spoken about positively and they encourage others to try it.

The goal of this project is to gain insights into what features are associated with higher view counts, so as to make recomendations to vegan content creators to optimize their reach.

## Objectives

- Scrape Youtube metadata using official API
- Prepare data for modeling
- Model to predict view count using R2 and RMSE as metrics
- Interpret model behaviour using SHAP and Partial Dependancy Plots
- Identify actionable insights for content creators

## Tools & technlogies

- **Language:** Python
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, shap, googleapiclient, os, re, ast
- **API:** Youtbe Data API v3
- **IDE:** Jupiter notebook (VS code)

## Data collection

In the first file [01_scraping.ipynb](/01_scraping.ipynb) the Youtube client object was built using the official Google python client library found [here](https://github.com/googleapis/google-api-python-client). An API key was then generated using a google account, and set as an environment variable. A video search was then done for key search terms such as "animal liberation" and "vegan speech", also for key animal rights content creators such as "joey carbstrong" and "earthling ed". Each search returned the video IDs for the first 100 results. The list of IDs was de-duplicated, and data on each individual video was collected, including the channel ID. As category returns an ID rather than the name, an additional search was used to collect the names. Finally the channel ID collected earlier was used to collect additional channel info for each video, such as subscribber and video count. All data was saved into the [/data](/data/) directory as [videos_unprepared.csv](/data/videos_unprepared.csv).

## Feature engineering

In [02_cleaning.ipynb](/02_cleaning.ipynb) I first extracted the hour, weekday and month of the year the video was published from the date. Next, as the category and channel data were in seperate csv files, I imported them and merged everything into one dataframe. The duration was initially in a string format e.g. "PT1H32M15S" for 1 hour, 32 mins and 15 seconds. I converted this into total seconds for a consistent numarical value. Binary variables Were turned into 1s and 0s, for definition 0 = sd & 1 = hd, and for caption 0 = False, 1 = True. I removed videos with the hastag #shorts in the title and also filtered out any channel names that had videos in the data that weren't representative of the target population e.g. they were talking about veganism but not promoting animal rights. Next I one-hot encoded the categories, dropping the dedundant first column which was "Autos & Vehicles". This gave me a feature column for every category with 0 for False and 1 for True.

For tags, as there were almost 5000 unique ones, I just created features for the top 11 and also a seperate one for total tag count.

![Top tags](/images/tag_frequency.png)

Next I created features for title and decription length, and also used a tf-idf vectorizer to vectorize the term importance in both, ignoring small words like "a" and "the". This created an extra 300 features of terms.

Checking numerical variable distributions, it can be seen that there is a strong right skew where the majority of variables like view and subscribber counts are at the lower end, and only a small minority go viral:

![Variable distributions](/images/variable_distributions.png)

Then applying a log transformation to these variables, and a cube root transformation to the description length as its skew is less extreeme. This is to normalize the data so that variables with larger scales don't dominate the results:

![Transformed distributions](/images/transformed_variable_distributions.png)

The prepared data was finally saved into the [/data](/data/) directory as [videos_prepared.csv](/data/videos_prepared.csv).

## Modeling

The target variable I am modelling to predict is the log transformed view count _log_view_count_. This means the predictions will need to be scaled back to to get in number of views. The first approach to this was to try and find a linear relationship between the features and the target. I standardized the data, as as some of the features have different ranges, and then applied a lasso regression for feature selection, to find what the most relevant features were for predicting view count. I tried multiple alpha values and looped through them to get the highest R2 and lowest root mean squared error (RMSE). Using the optimal alpha value of 0.1, the features were reduced from 337 to 25. A linear model was fitted to the training data for just these features and then on the test data using these features the R2 was about 0.53 and RMSE about 1.81 (log).

Looking at the models top coefficients, subscriber count has the strongest association in explaining the variance in view count, followed by the channels total view count and then whether or not is of the Music category. All of the top features appear to have a positive correlation with the target, with the exception of the term "surge" which is negative:

![Linreg features](/images/linreg_top_features.png)

The residuals for the predictions were then plotted. Looking at this, we can see that at the low prediction levels up to log 6, the model tended to over predict, amost by as much as log 8. Then at the mid prediction levels of about log 6 to 12 the residuals seem evenly spread out, but then above log 12 all residuals appear to be negative and so the model is overpredicting, although not as extreeme as the underpredictions at the lower end.

![Linreg residuals](/images/linreg_residual_plot.png)

This pattern in residuals at varying levels of prediction suggests a more flexible model may be appropriate, so next I tried a decision tree. For this I used hyperparameter tuning to find the optimal parameters, and fitted a model with these to the training data. The R2 for the decision tree on the test data was about 0.63, and RMSE was about 1.59. An improvement, but tried to see if I could do better using a random forest. This time for hyperparameter tuning I used RandomSearchCV to search a wide range of randomly selected parameters, and then using the best parameter results, used GridSearchCV to fine-tune a narrower range. Once I fit the random forest with the optimal parameters and scored it against the test data, the R2 was about 0.73 and RMSE about 1.38, a big improvement from the original linear model.

There doesn't appear to be any clear pattern for the residuals of the random forest, indicating the model is well-fitted.

![RandForest residuals](/images/rf_residual_plot.png)

The top features for the random forest subscribber, channel video & channel view count:

![Randforest features](/images/rf_top_feature_importance.png)

## Interpretation

### Linear regression [03_modeling_linreg.ipynb](/03_modeling_linreg.ipynb)

When sorted by log_channel_sub_count the top coefficient we can see 'TEDx Talks' is the most subscribed to channel and that there are 5 videos all with high views, the most popular being 'Every Argument Against Veganism' by Ed Winters with a view count when scaled back of over 2.34 million. Other high subscriber, high videos included another 'TEDx Talks' by Moby, A 'Big Think' video with Peter Singer, two 'Brut India' videos with Joaquin Phoenix, and an 'Oxford Union' video with Joey Carbstrong. This suggets a pattern of high subscriber non-vegan channels having big name people, whether known for veganism or other things, on their channel for speeches/talks/interviews is a good indicater of higher view count. When sorting by log_channel_view_count we get a lot of the same videos including 'TEDx Talks', 'Big Think' & 'Brut India' suggesting subscribers and channel views are strongly correlated. Sorting by Music category, we can see there some high view songs, which is why the model determined that important. The highest being 'ALO (Animal Liberation Orchestra) - Girl, I Wanna Lay You Down ft. Jack Johnson' with over 501k views.

Looking at the regression line of the top three coefficients we can see that subscribber and channel view count both have a strong correlation with video view count. Music category on the other hand, despite being the third strongest coefficient has a flat regression line. This may be because there are other features that music videos tend to be correlated with that have a negative impact on view count.

![Linreg coef reg lines](/images/top_coefficient_regression_lines.png)

### Random forest [04_modeling_dt_rf.ipynb](/04_modeling_dt_rf.ipynb)

Using a partial dependence plot we can see that keeping all other features constant, there is a strong positive trend between subscriber count and view count. This trend is gradual at first, but seems to shoot off around 20k+ subscribers. The trend starts to slow down again around 160k subscribers. There is also a positive trend for channel total view count, although much weaker. In fact there is a large interval between about 440k and 65.6 million channel views, where this feature doesn't fluctuate in its influance for predicting video views. For total video count of the channel, suprisingly there is a negative trend. The plot shows a mostly consistent trend downwards, meaning that with all other features held equal, the more videos a channel has, the fewer views it is predicted to have.

![pdp1](/images/pdp_sub_video_channel_view_count.png)

Looking at the duration, there is a non-linear relationship between views, the general trend is positive, however there is a large dip around 53 seconds that starts to increase again around 1.5 mins. This is probably where there are still some Youtube shorts still included in the data. After this the trend is positive until about 40.5 mins when the influence starts to dip again. Description length initially has a negative influence on view count up to about 125 characters all the way up to about 1300 characters. Higher tag counts also appar to be associated with more views, with 18 tags being the peak, it starts to dip down a little again after that.

![pdp2](/images/pdp_duration_desc_length_tag_count.png)

Publishing a video between 12 and 1am there is a clear stronger influence on view count than any other time of day. As for day of the week, there is a dip in influence on Monday and Tuesday, but all other days are equal. Finally all months of the year appear to be equal, with the exception of January which is the weakest.

![pdp3](/images/pdp_hour_month_weekday_pub.png)

The beeswarm shows the shap values of the most important terms used in the video titles and descriptions. Most notably the term "outreach" has a strong negative impact on view count. This could be because outreach videos are more of a niche activism focused genre, it could also be correlated with other features that are having a negative impact. The term "music" has a positive impact.

![Term beeswarm](/images/term_beeswarm.png)

While most categories in the dataset have little to no impact on view count, the strongest negative impact is Science & Technology, and the strongest positive is Music, which is consistent with what we saw on the term beeswarm. Categories such as People & Blogs, Howto & Style and Education all had some positive impact as well.

![Category beeswarm](/images/category_beeswarm.png)

## Further investigation [05_investigation_shap.ipynb](/05_investigation_shap.ipynb)

### Outreach

Since containing the term "outreach" had a negative global impact in the beeswarm, I decided to look at the local impact of the highest viewed outreach videos using waterfall plots. Looking at these we can see they are all getting most of their negative impact from subscriber count, and sometimes channel view count. Channel video count is the main positive impact on views for these videos, which as we saw from the pdp earlier video count has a negative global influence, suggesting that channels that upload outreach videos generally have fewer videos on average.

From this we can see that although there is some small negative signal from containing the term "outreach", for the most part they aren't being inherently penalized, but rather are coming from smaller, less established channels with less subscribers and channel views. Possible recommendations: A/B test alternative titles that omit the word "outreach" entirely to see if there's any effect on view count, or mix outreach with broader categories, or do collabs to build subscribers.

- [Vegan Outreach 485](/images/local_shap/vegan_outreach_485.png)
- [James Kite 671](/images/local_shap/james_kite_671.png)
- [James Kite 464](/images/local_shap/james_kite_464.png)
- [Clif Grant 380](/images/local_shap/clif_grant_380.png)
- [The Cranky Vegan](/images/local_shap/the_cranky_vegan_683.png)
