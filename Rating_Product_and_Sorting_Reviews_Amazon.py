
###################################################
# PROJECT: Rating Product & Sorting Reviews in Amazon
###################################################

# http://jmcauley.ucsd.edu/data/amazon/links.html

# reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
# asin - ID of the product, e.g. 0000013714
# reviewerName - name of the reviewer
# helpful - helpfulness rating of the review, e.g. 2/3
# reviewText - text of the review
# overall - rating of the product
# summary - summary of the review
# unixReviewTime - time of the review (unix time)
# reviewTime - time of the review (raw)


###################################################
# Calculate the ratings of a product according to recent reviews adn compare with old ratings.
###################################################
import pandas as pd
import numpy as np
import math
import scipy.stats as st
import datetime as dt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# Step 1. Download data
###################################################

# http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics.json.gz

###################################################
# Step 2. find most reviewed product
###################################################

# B007WTAJTO
# df_sub data asin

###################################################
# Step 3. Reduce dataset according to most reviewed product (df_sub)
###################################################

# df_sub

df_ = pd.read_csv("Datasets/df_sub.csv")
df = df_.copy()
df.head()
df.shape
# (4915, 9)

###################################################
# Step 4. Average score of the product?
###################################################

df["overall"].mean()
# 4.587589013224822

df["overall"].quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
df["overall"].describe()


###################################################
# Step 5. Weighted score average according to date
###################################################

# day_diff calculation: (how many days passed after the review)
df['reviewTime'] = pd.to_datetime(df['reviewTime'], dayfirst=True)
current_date = pd.to_datetime('2014-12-08 0:0:0')
df["day_diff"] = (current_date - df['reviewTime']).dt.days

# Divide time to quarterly values:
q1 = df["day_diff"].quantile(0.25) # 281.0
q2 = df["day_diff"].quantile(0.50) # 431
q3 = df["day_diff"].quantile(0.75) # 601

df["day_diff"].max() # 1064

df.loc[df["day_diff"] <= q1, "overall"].mean() * (28/100) +\
    df.loc[(df["day_diff"] > q1) & (df["day_diff"] <= q2), "overall"].mean() * (26/100) +\
    df.loc[(df["day_diff"] > q2) & (df["day_diff"] <= q3), "overall"].mean() * (24/100) +\
    df.loc[df["day_diff"] > q3, "overall"].mean() * (22/100)

# 4.595593165128118

df.head()
###################################################
# Find the first 20 review which should be shown on the product description page.
###################################################

###################################################
# Step 1. "Helpful" variable --> 1: helpful_yes, 2: helpful_no,  3: total_vote
###################################################

# Helpful : 1- # of votes who found the review helpful 2- # of total votes.
# (total_vote - helpful_yes) --> helpful_no

df["helpful"].head()

abc = df["helpful"].str.strip("[]")
abc2 = abc.str.split(",", expand = True)
abc2.head()
abc2.columns = ["helpful_yes", "total_vote"]

df.head()
type(abc2)
df["helpful_yes"] = abc2["helpful_yes"]
df["total_vote"] = abc2["total_vote"]
df["helpful_yes"] = df["helpful_yes"].astype(int)
df["total_vote"] = df["total_vote"].astype(int)
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

type(df["helpful_yes"][0])


# alternativ
df["helpful_yes"] = df[["helpful"]].applymap(lambda x: x.split(",")[0].strip("[")).astype(int)
df["total_vote"] = df[["helpful"]].applymap(lambda x: x.split(",")[1].strip("]")).astype(int)
df.head()
###################################################
# Step 2. create scores according to "score_pos_neg_diff" and save it to df_sub
###################################################


def score_pos_neg_diff(pos, neg):
   score = pos - neg
   return score


df["score_pos_neg_diff"] = score_pos_neg_diff(df["helpful_yes"], df["helpful_no"])
df.head()

###################################################
# Step 3. create scores according to "score_average_rating" and save it to df_sub
###################################################


def score_average_rating(pos, neg):
    if pos + neg == 0:
        return 0
    return pos / (pos + neg)


df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],x["helpful_no"]), axis=1)

##################################################
# Step 4. create scores according to "wilson_lower_bound" and save it to df_sub
###################################################


def wilson_lower_bound(pos, neg, confidence=0.95):
    """
    Wilson Lower Bound Score calculation

    - The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is accepted as the WLB score.
    - The score to be calculated is used for product ranking.
    - Note: If the scores are between 1-5, they are marked as 1-3 downs, 4-5 ups and matched to bernoulli.

    Parameters
    ----------
    pos: int
        number of positive comments
    neg: int
        number of negative comments
    confidence: float
        confidence interval

    Returns
    -------
    wilson score: float

    """
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2) # t-table, z-score
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head()

##################################################
# Step 5. Find the first 20 reviews to show in the product description page
###################################################

# The dataset is sorted according to the data obtained on whether the comments are useful or not.:

df[["reviewText", "overall", "score_pos_neg_diff","score_average_rating", "wilson_lower_bound" ]].sort_values("score_pos_neg_diff", ascending=False).head(20)


# In this ranking, the "score_average_rating" score appears to be more significant than just the difference between the interpretations.
# Comments with a high score_average_rating value are lower and lower ones are higher.
# Sorting by score_pos_neg_diff is not meaningful.

df[["reviewText", "overall", "score_pos_neg_diff","score_average_rating", "wilson_lower_bound" ]].sort_values("score_average_rating", ascending=False).head(20)
# When we sort according to score_average_rating, this time it is seen that high data in wilson lower bound values remain below the ranking.
# but the result of this ranking is not meaningless either.

df[["reviewText", "overall", "score_pos_neg_diff","score_average_rating", "wilson_lower_bound" ]].sort_values("wilson_lower_bound", ascending=False).head(20)
# When we sort by wilson lower bound, it has been observed that the values of 1 in the score_average_rating data decrease.
# While there were only 2 comments that gave a rating of 2 in the top 20; After wlb, the comments that gave a rating of 1 also rose to the top.

# It will be healthy if the first 20 comments to be displayed on the site are in the wlb ranking.

#                                              reviewText  overall  \
# 2031  [[ UPDATE - 6/19/2014 ]]So my lovely wife boug...  5.00000
# 3449  I have tested dozens of SDHC and micro-SDHC ca...  5.00000
# 4212  NOTE:  please read the last update (scroll to ...  1.00000
# 317   If your card gets hot enough to be painful, it...  1.00000
# 4672  Sandisk announcement of the first 128GB micro ...  5.00000
# 1835  Bought from BestBuy online the day it was anno...  5.00000
# 3981  The last few days I have been diligently shopp...  5.00000
# 3807  I bought this card to replace a lost 16 gig in...  3.00000
# 4306  While I got this card as a "deal of the day" o...  5.00000
# 4596  Hi:I ordered two card and they arrived the nex...  1.00000
# 315   Bought this card to use with my Samsung Galaxy...  5.00000
# 1465  I for one have not bought into Google's, or an...  4.00000
# 1609  I have always been a sandisk guy.  This cards ...  5.00000
# 4302  So I got this SD specifically for my GoPro Bla...  5.00000
# 4072  I used this for my Samsung Galaxy Tab 2 7.0 . ...  5.00000
# 1072  What more can I say? The 64GB micro SD works f...  5.00000
# 2583  I bought this Class 10 SD card for my GoPro 3 ...  5.00000
# 121   Update: providing an update with regard to San...  5.00000
# 1142  As soon as I saw that this card was announced ...  5.00000
# 1753  Puchased this card right after I received my S...  5.00000
#       score_pos_neg_diff  score_average_rating  wilson_lower_bound
# 2031                1884               0.96634             0.95754
# 3449                1351               0.94884             0.93652
# 4212                1442               0.92562             0.91214
# 317                  349               0.85253             0.81858
# 4672                  41               0.91837             0.80811
# 1835                  52               0.88235             0.78465
# 3981                  85               0.80576             0.73214
# 3807                  19               0.88000             0.70044
# 4306                  37               0.78462             0.67033
# 4596                  55               0.75229             0.66359
# 315                   28               0.79167             0.65741
# 1465                   7               1.00000             0.64567
# 1609                   7               1.00000             0.64567
# 4302                  12               0.87500             0.63977
# 4072                   6               1.00000             0.60967
# 1072                   5               1.00000             0.56552
# 2583                   5               1.00000             0.56552
# 121                    5               1.00000             0.56552
# 1142                   5               1.00000             0.56552
# 1753                   5               1.00000             0.56552





