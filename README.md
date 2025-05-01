The paper "What’s in a name? Understanding the Interplay between Titles, Content, and Communities in Social Media" by Lakkaraju, McAuley, and Leskovec (ICWSM 2013) investigates how titles (post headlines), content, and subreddit communities interact to influence the popularity of posts on Reddit.
📄 Summary of the Paper
Goal:
To understand how the title and content of Reddit posts — along with the subreddit where they are posted — affect post popularity (measured by upvotes).

🧪 Dataset Used
Source: Reddit data (primarily from 2008–2012)

Size: Approximately 132,000 Reddit posts

Scope: Focused on submissions (not comments) from 20 diverse subreddits (e.g., r/pics, r/science, r/politics)

🔍 Preprocessing & Feature Extraction
Data Collection:

Posts were gathered using Reddit APIs and included metadata like upvotes, subreddit, title, and content.

Filtering: Removed posts with no content or titles, and filtered by post length and subreddit activity.

Text Preprocessing: Tokenized and cleaned titles and content.Removed stopwords and applied stemming

Feature Engineering: Title features: Length, sentiment, readability, use of slang, etc. Content features: Similar metrics as above, plus topic modeling via LDA Subreddit interaction: Modeled how post characteristics interacted differently in each subreddit

📊 Key Findings & Conclusions
Titles Matter a Lot:
The title alone can predict post popularity almost as well as the title+content combined. This suggests that users often decide to engage with a post based on the headline.

Subreddit Norms Are Crucial:
Different communities have different preferences. A title or tone that works well in one subreddit may fail in another.

Topic Sensitivity Varies:
Some subreddits are more sensitive to the content’s actual topic, while others respond more to stylistic features (e.g., humor, emotional tone).

Modeling Community-Specific Preferences Improves Prediction:
Custom models trained per subreddit significantly outperformed global models.


💡 Visualization Ideas We Can Do with Reddit Data (Inspired by the Paper)
->Title Length vs. Upvotes

Scatter or line plot showing how title length correlates with popularity across different subreddits.

->Emotion vs. Popularity

Use sentiment analysis (e.g., VADER) to plot sentiment polarity of titles vs. upvotes.

->Topic Trends by Subreddit

Apply topic modeling (LDA) and visualize common topics over time or subreddit (e.g., heatmap).

->Title Readability

Compute readability scores (e.g., Flesch-Kincaid) and plot them against post success.

->Subreddit-Specific Word Clouds

Generate word clouds of most successful vs. least successful post titles in each subreddit.

->Predictive Modeling

Train a simple model (e.g., logistic regression or random forest) to predict if a post will get more than X upvotes based on its title and subreddit.

