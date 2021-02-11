# Dataset Analysis
- 44.8k observations (~21k True, ~23k False)
- labeled data (0 or 1, fake news or true)
- kaggle (not competition): https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset 
- two text columns: title, and subject with lots of text
- one categorical column with 8 distinct categories of news
  - categories are not equally represented
  - category 1 has 3x the data as the sum of the last 3
  - top 2 categories account for almost 50% of the data
- one date column shows data is 1-3 years old 
- easy to import and vectorize (already done)

# Data Questions
- Can we predict fake news by just the title?
- Can we predict fake news by title + subject?
- Which category is easiest to predict news for?
  - proportionality of categories would need to be taken into account, of course
- Does punctuation make a noticeable difference in prediction?
  - Are quotes indicative of "real" titles?
  - Do question-marks imply false news?
