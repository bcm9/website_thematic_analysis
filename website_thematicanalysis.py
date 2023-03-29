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

# Import questionnaire results
df = pd.read_excel('C:\\Users\\bcm9\\Documents\\Py_code\\ThematicAnalysis\\website_questionnaire.xlsx')

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
# Codes selected based on the research question: generate list of codes that capture concepts and themes. Codes should be grounded in the data and reflect experiences, perspectives, language. 
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
# Bar plot of code frequency
code_counts = df.iloc[:, 3:].sum()
plt.bar(code_counts.index, code_counts.values)
plt.xlabel("Code", fontweight='bold')
plt.ylabel("Frequency", fontweight='bold')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.75)
plt.show()

# Plot histogram of ratings
plt.hist(df['Rating'], bins=[1, 2, 3, 4, 5, 6], color='steelblue')
plt.xlabel('Rating', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.grid(axis='y', alpha=0.75)
plt.show()