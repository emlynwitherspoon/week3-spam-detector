# Week 3 Email Spam Detector — Analysis Report

## 3.6 What Is Wrong With My Data?

My dataset contains only 16 emails, which is too small to train a reliable spam detector. I found 2 duplicate rows and 1 overlapping row between the training and testing sets. This means the model may have already seen some of the test data during training. Although my model achieved 100% accuracy, I cannot fully trust that result because the dataset is small and contains duplicate information. I would need more unique emails and a better separation of training and testing data to evaluate the model properly.

## Features Used

I used six features to help my model identify spam emails: word count, exclamation marks, money-related words, capital letters, urgent words, and question marks.

I chose these features because spam emails often use exciting language, money offers, capital letters, and urgent messages to get people's attention. These features can help the model recognize patterns that may indicate spam.

## Model Results

My spam detector achieved 100% accuracy, 100% precision, and 100% recall on the test set.

The confusion matrix was:

[[2, 0],
 [0, 2]]

This means the model correctly identified 2 legitimate emails and 2 spam emails. It made no mistakes on these four test emails.

However, because the dataset is very small and there was overlap between the training and testing data, these results do not prove that the model will perform perfectly on real emails.

## Which Error Matters More?

I believe a false positive is more serious than a false negative. A false positive happens when a legitimate email is incorrectly marked as spam. This could cause someone to miss an important message from work, school, or a doctor.

A false negative happens when a spam email is incorrectly marked as legitimate. Although this is also a problem, I would rather receive an occasional spam email than lose an important legitimate message.

I would adjust the spam filter to be more careful before marking emails as spam.

## What the Model Learned

The feature weights showed that exclamation marks had the strongest influence on my spam detector, with a weight of 1.1749. Money-related words came next with a weight of 0.8133, followed by urgent words at 0.4995.

This mostly matches what I expected because spam emails often use lots of exclamation marks, money offers, and urgent language to get people's attention.

I was surprised that exclamation marks had more influence than money-related words. This shows that the model learned to rely heavily on punctuation when identifying spam. However, this could cause problems because legitimate emails can also contain exclamation marks.