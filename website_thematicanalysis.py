# -*- coding: utf-8 -*-
"""

Analysis of website feedback (questionnaire with free text and ratings)

Thematic analysis aims to identify the underlying patterns and themes in qualitative data and draw meaningful conclusions

"""
###############################################################################################################
# Import libraries
import nltk as nltk # for NLP tasks
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# Import questionnaire results
df = pd.read_excel('C:\\Users\\bc22\\OneDrive\\Documents\\code\\website_thematic\\website_questionnaire.xlsx')

###############################################################################################################
# Preprocess the feedback with WordNetLemmatizer
# NLP tool reduces words to lemma (base/dictionary form, e.g. dances/dancing = dance)

# Download packages if necessary
#nltk.download('punkt')
#nltk.download("stopwords")
#nltk.download('omw-1.4')
#nltk.download('wordnet')

# Set stop words to filter out
#stopwords = set(nltk.corpus.stopwords.words("english"))
custom_stopwords = set(["the", "and", "a", "an", "to", "in", "of", "for"])

# Import WordNetLemmatizer from the nltk.stem module
from nltk.stem import WordNetLemmatizer

# Create WordNetLemmatizer object
lemmatizer = WordNetLemmatizer()

# Create preprocessing function with custom stop words and lemmatizer
def preprocess_text(text):
    # Tokenise text
    tokens = nltk.word_tokenize(text)
    # Remove custom stop words, lemmatize remaining words
    words = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in custom_stopwords]
    # Join words back into a single string
    return " ".join(words)

# Apply to feedback col
df["Feedback Response Processed"] = df["Feedback Response"].apply(preprocess_text)

###############################################################################################################
# Create initial codes using regexps
# Themes and codes human led, selected based on the research question: generate list of codes that capture concepts and themes. 
# Codes should be grounded in the data and reflect experiences, perspectives, language. 
# Process should be systematic and transparent: ensure codes are reliable, valid, and consistent
codes = {"Design": ["design", "layout", "color scheme"],
         "Performance": ["slow", "crash", "load"],
         "Usability": ["use", "navigat", "clutter", "search", "function"],
         "Descriptions": ["service", "description"]}

###############################################################################################################
# Nested for loop iterates over feedback, checks for each keyword in current code
# Resulting True/False stored in a list for each code, added as a new col to df
for code, keywords in codes.items():
    # Create empty list
    code_values = []
    # Loop through the feedback responses
    for response in df["Feedback Response Processed"]:
        # Check if any of the keywords for the current code are present in the response
        keyword_present = False
        for keyword in keywords:
            if re.search(keyword, response):
                keyword_present = True
                break # Exit loop
        # Append True or False to list based on whether the code found in the response
        code_values.append(keyword_present)
    # Add list of values as new col in df
    df[code] = code_values
    
###############################################################################################################
# Plotting results
fs=16
xtfs=14
plt.rcParams['font.family'] = 'Calibri'

# Bar plot of code frequency
plt.figure(figsize=(8, 6))
code_counts = df.iloc[:, 3:].sum()
plt.bar(code_counts.index, code_counts.values, color='skyblue', edgecolor='black')
plt.xlabel("Themes", fontweight='bold', fontsize=fs)
plt.ylabel("Frequency", fontweight='bold', fontsize=fs)
plt.xticks(rotation=45, fontsize=xtfs)
plt.yticks(fontsize=xtfs)
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()

# Plot histogram of ratings
plt.figure(figsize=(8, 6))
plt.hist(df['Rating'], bins=np.arange(0.5, 6.5, 1), color='skyblue', edgecolor='black')
plt.xlabel('Rating', fontweight='bold', fontsize=fs)
plt.ylabel('Frequency', fontweight='bold', fontsize=fs)
plt.title("Distribution of Ratings", fontweight='bold', fontsize=fs+2)
plt.xticks(fontsize=xtfs)
plt.yticks(fontsize=xtfs)
plt.grid(axis='y', alpha=0.75)
plt.show()

# Word cloud plot
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Concatenate all the processed responses into a single string
text = " ".join(df["Feedback Response Processed"].values)
# Create a word cloud object
wordcloud = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(text)
# Display word cloud
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

###############################################################################################################
# Sentiment analysis
from textblob import TextBlob

# Define function to calculate the sentiment polarity of text
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return sentiment

# Apply sentiment analysis function to the processed column
df["Sentiment"] = df["Feedback Response Processed"].apply(get_sentiment)

# Count number of positive and negative responses
num_pos = (df["Sentiment"] > 0).sum()
num_neg = (df["Sentiment"] < 0).sum()

# Print results
print(f"Positive responses: {num_pos}")
print(f"Negative responses: {num_neg}")

# Plot bar chart
fig, ax = plt.subplots(figsize=(7, 6))
ax.bar(["Positive", "Negative"], [num_pos, num_neg], color=["#4CAF50", "#F44336"])
ax.set_xticklabels(["Positive", "Negative"], fontsize=fs)
ax.set_xlabel("Sentiment", fontsize=fs, fontweight="bold")
ax.set_ylabel("Number of Responses", fontsize=fs, fontweight="bold")
ax.set_title("Sentiment Analysis", fontsize=fs+2, fontweight="bold")
plt.grid(axis='y', alpha=0.75)
plt.show()