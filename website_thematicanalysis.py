# -*- coding: utf-8 -*-
"""

NLP-based analysis of website feedback (questionnaire with free text and ratings)


"""
###############################################################################################################
# Import libraries
import nltk as nltk # for NLP tasks
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import numpy as np
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

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
custom_stopwords = set(["the", "and", "a", "an", "to", "in", "of", "for", "is", "i", "it"])

# Update your list of stopwords to include NLTK's stopwords, your custom stopwords, and punctuation
all_stopwords = set(stopwords.words('english')).union(custom_stopwords, set(string.punctuation))

# Import WordNetLemmatizer from the nltk.stem module
from nltk.stem import WordNetLemmatizer

# Create WordNetLemmatizer object
lemmatizer = WordNetLemmatizer()

# Create preprocessing function with custom stop words and lemmatizer
def preprocess_text(text):
    # Tokenise text
    tokens = nltk.word_tokenize(text)
    # Remove custom stop words, lemmatize remaining words
    words = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in all_stopwords]
    # Join words back into a single string
    return " ".join(words)

# Apply to feedback col
df["Feedback Response Processed"] = df["Feedback Response"].apply(preprocess_text)

#############################################################################################################################
# Thematic analysis
# Aims to identify the underlying patterns and themes in qualitative data and draw meaningful conclusions
#############################################################################################################################
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
fs=20
xtfs=fs-2
plt.rcParams['font.family'] = 'Calibri'
# plt.rcParams['font.family'] = 'Circular Book'

# Bar plot of code frequency
plt.figure(figsize=(7, 6))
code_counts = df.iloc[:, 3:].sum()
plt.bar(code_counts.index, code_counts.values, color='skyblue', edgecolor='white', alpha=0.8)
plt.xlabel("Themes", fontweight='bold', fontsize=fs)
plt.ylabel("Frequency", fontweight='bold', fontsize=fs)
plt.title("Distribution of Themes", fontweight='bold', fontsize=fs+2)
plt.xticks(rotation=45, fontsize=xtfs)
plt.yticks(fontsize=xtfs)
plt.grid(axis='y', alpha=0.35)
plt.tight_layout()
# Remove axes borders
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
# Remove x and y tick lines
plt.tick_params(axis='both', which='both', length=0)
# Save and show the histogram
folder = 'C:/Users/bc22/OneDrive/Documents/code/website_thematic/'
plt.savefig(folder+'figure1.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot histogram of ratings
plt.figure(figsize=(8, 6))
plt.hist(df['Rating'], bins=np.arange(0.5, 6.5, 1), color='skyblue', edgecolor='white', alpha=0.8)
plt.xlabel('Rating', fontweight='bold', fontsize=fs)
plt.ylabel('Frequency', fontweight='bold', fontsize=fs)
mean_rating = np.mean(df['Rating'])
median_rating = np.median(df['Rating'])
std_rating = np.std(df['Rating'])
q75, q25 = np.percentile(df['Rating'], [75 ,25])
iqr_rating = q75 - q25
plt.title(f"Distribution of Ratings (Med = {median_rating:.1f}; IQR = {iqr_rating:.1f})", fontweight='bold', fontsize=fs+2)  # Format mean to 2 decimal places
plt.xticks(fontsize=xtfs)
plt.yticks(fontsize=xtfs)
plt.grid(axis='y', alpha=0.25)
# Remove axes borders
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
# Remove x and y tick lines
plt.tick_params(axis='both', which='both', length=0)
plt.savefig(folder+'figure2.png', dpi=300, bbox_inches='tight')
plt.show()

# Word cloud plot
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create a circular mask
x, y = np.ogrid[:800, :800]
mask = (x - 400) ** 2 + (y - 400) ** 2 > 390 ** 2  # 390 is the radius, adjust as needed
mask = 255 * mask.astype(int)  # 255 makes the mask white, which wordcloud recognises
font_path = 'C:/Windows/Fonts/calibri.ttf'  # Adjust the path based on where Calibri is located on your system


# Concatenate all the processed responses into a single string
text = " ".join(df["Feedback Response Processed"].values)
# Create a word cloud object
# wordcloud = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(text)
wordcloud = WordCloud(
    width=800, 
    height=800, 
    background_color='white', 
    colormap='viridis',  # Colour map for aesthetic appeal
    max_words=100,       # Adjust max words to control density
    min_font_size=20,    # Minimum font size for smaller words
    max_font_size=150,   # Maximum font size for larger words
    random_state=42,     # Random state for reproducibility
    collocations=False,  # Avoid bigram grouping
    contour_width=.4,     # Adds a contour for emphasis
    contour_color='white',  # Contour colour for better visibility
    mask=mask, # Apply the circular mask
    font_path=font_path,  # Set the Calibri font
).generate(text)
# Display word cloud
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig(folder+'figure3.png', dpi=300, bbox_inches='tight')
plt.show()

###############################################################################################################
# Latent Dirichlet Allocation (LDA) 
# Probabilistic model to identify latent topics within a collection of documents
###############################################################################################################
# Function to split the preprocessed text into tokens
def split_into_tokens(text):
    return text.split()

# Apply the function to the 'Feedback Response Processed' column
df['Tokens'] = df['Feedback Response Processed'].apply(split_into_tokens)

# Now we can create the dictionary and corpus needed for LDA
dictionary = corpora.Dictionary(df['Tokens'])
corpus = [dictionary.doc2bow(text) for text in df['Tokens']]

# Apply LDA
ldamodel = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Print topics
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)

# Adding the dominant topic for each document to the dataframe
dominant_topics = []
for text_corpus in corpus:
    topics_in_text = ldamodel[text_corpus]
    dominant_topic = sorted(topics_in_text, key=lambda x: x[1], reverse=True)[0][0]
    dominant_topics.append(dominant_topic)

# Output of n topics extracted from a dataset
# Each topic described by a set of words with corresponding probabilities
# Numbers indicate likelihood that word is associated with topic, suggesting how representative it is
df['Dominant_Topic'] = dominant_topics

###############################################################################################################
# Sentiment analysis
###############################################################################################################
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

# Binomial test: whether the proportion of positive sentiments is significantly different from 50%
from scipy.stats import binom_test
# Null hypothesis: p = 0.5 (the probability of success in a single trial)
p_value = binom_test(num_pos, n=num_pos+num_neg, p=0.5)

# Plot bar chart
fig, ax = plt.subplots(figsize=(7, 6))
ax.bar(["Positive", "Negative"], [num_pos, num_neg], color=["#00A36C", "#FA5F55"], alpha=0.8)
ax.set_xticklabels(["Positive", "Negative"], fontsize=fs)
ax.set_xlabel("Sentiment", fontsize=fs, fontweight="bold")
ax.set_ylabel("Number of Responses", fontsize=fs, fontweight="bold")
ax.set_title(f"Sentiment Analysis (p = {p_value:.4f})", fontsize=fs+2, fontweight="bold")
plt.xticks(fontsize=xtfs)
plt.yticks(fontsize=xtfs)
plt.grid(axis='y', alpha=0.25)
# Remove axes borders
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
# Remove x and y tick lines
plt.tick_params(axis='both', which='both', length=0)
plt.savefig(folder+'figure4.png', dpi=300, bbox_inches='tight')
plt.show()