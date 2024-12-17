import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

text = open('read.txt', encoding='utf-8').read()
lower_text = text.lower()
cleaned_text = lower_text.translate(str.maketrans('', '', string.punctuation))
tokenized_words = word_tokenize(cleaned_text, 'english')

final_words = []

for word in tokenized_words:
    if word not in stopwords.words('english'):
        final_words.append(word)

emotion_list = []
with open('emotion.txt', 'r') as file:
    for line in file:
        clear_line = line.replace('\n', '').replace(',', '').replace('\'', '').strip()
        word, emotion = clear_line.split(':')
        if word in final_words:
            emotion_list.append(emotion)

emotion_counter = Counter(emotion_list)
def sentiment_analyzer(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    negative, positive, neutral = score['neg'], score['pos'], score['neu']
    
    if negative > positive:
        print('Negative Sentiment')
    elif positive > negative:
        print('Positive Sentiment')
    else:
        print('Neutral Sentiment')

sentiment_analyzer(cleaned_text)