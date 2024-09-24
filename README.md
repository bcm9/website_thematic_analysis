# website_thematic_analysis

- I built a small website for my team and wanted to get some feedback on it. I did this using a simple questionnaire of free text and ratings.
- This code involves a thematic analysis, latent Dirichlet allocation and sentiment analysis of this feedback to identify themes, topics and patterns. The analysis was carried out in accordance with Braun and Clarke's (2006) guidelines.
  - The NLTK Python library was used for natural language processing, and the Pandas library was used for data manipulation.
  - The feedback was pre-processed using the WordNetLemmatizer to reduce words to their lemma, and a custom set of stopwords was used to filter out irrelevant words.
  - This work also includes a sentiment analysis of the free text, classifing the sentiment polarity of a text to positive or negative.
  - A binomial test is used to check whether the number of positive sentiments is significantly different from 50%.


<div style="display: flex; justify-content: space-between;">
  <img src="./Figures/figure1.png" alt="themes" width="420"/>
  <img src="./Figures/figure2.png" alt="ratings" width="420"/>
  <img src="./Figures/figure4.png" alt="sentiment" width="420"/>
  <img src="./Figures/figure3.png" alt="word cloud" width="420"/>
</div>
