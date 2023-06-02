import numpy as np
import os
from django.conf import settings

scaler_path = os.path.join(settings.STATIC_ROOT, 'scaler.pickle')
model_path = os.path.join(settings.STATIC_ROOT, 'model')

def get_book_data(text, book_location): 
    import nltk

    from nltk.corpus import stopwords
    from nltk.corpus import state_union
    from nltk.tokenize import word_tokenize
    from nltk.tokenize import sent_tokenize
    from nltk.tokenize import RegexpTokenizer
    from nltk.tokenize import PunktSentenceTokenizer
    import nltk.data

    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment import SentimentIntensityAnalyzer


    if book_location: 
        with open(book_location, "r", encoding="utf8") as f: 
            text = f.read()
        #print(text)
        text = text.replace("\n"," ")
        text = text.replace("  ", " ")


    tokenizer = RegexpTokenizer(r'\w+')
    word_tokens = tokenizer.tokenize(text)

    stop_words = set(stopwords.words("english"))

    filtered_words = [w for w in word_tokens if w.lower() not in stop_words]
    
    
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sent_tokens=tokenizer.tokenize(text)
    
    
    
    # initialize a lemmatizer and lemmatize all words
    lemmatizer = WordNetLemmatizer()

    lemmatized_words=[]
    for word in filtered_words:
        word2=lemmatizer.lemmatize(word)
        pos_tagged=nltk.pos_tag(nltk.word_tokenize(word))
        if word!=word2:
            lemmatized_words.append(word2)
        else:
            try:
                lemmatized_words.append(lemmatizer.lemmatize(word, pos="a"))
            except:
                lemmatized_words.append(word2)

    #VADER            
    sia = SentimentIntensityAnalyzer()
    df_list = []
    df_sent_list=[]

    for word in lemmatized_words: 
        word_info = {}
        word_info["word"] = word
        word_info.update(sia.polarity_scores(word))
        df_list.append(word_info)

    for sentence in sent_tokens:
        sent_info={}
        sent_info["sentence"]=sentence
        sent_info.update(sia.polarity_scores(sentence))
        df_sent_list.append(sent_info)


    positive_sentence=[]
    def pos_seperator(self):
        x = sia.polarity_scores(self)['compound']
        if x>0:
            positive_sentence.append(x)

    negative_sentence=[]
    def neg_seperator(self):
        x = sia.polarity_scores(self)['compound']
        if x<0:
            negative_sentence.append(x)

    for sentence in sent_tokens: 
        neg_seperator(sentence)    
    
    for sentence in sent_tokens:      
        pos_seperator(sentence)   


    positive_sentence_sum = []
    for item in positive_sentence:
        positive_sentence_sum.append(np.power(item, 20))

    negative_sentence_sum = []
    for item in negative_sentence:
        negative_sentence_sum.append(np.power(item, 20))

    sum_neg = np.sum(negative_sentence_sum) 
    sum_pos = np.sum(positive_sentence_sum)  

    total_sum = sum_neg + sum_pos

    percentage_neg = sum_neg *100 / total_sum
    percentage_pos = sum_pos *100 / total_sum
    

    positive_words=[]
    negative_words=[]

    number_of_Positive_Sentences=len(positive_sentence)
    number_of_Negative_Sentences=len(negative_sentence)
    number_of_total_sentences= len(sent_tokens)
    number_of_total_words= len(word_tokens)
    number_of_words_per_sentences= number_of_total_words/number_of_total_sentences



    for j in range(len(df_list)):
        if df_list[j]['pos'] > 0:
            positive_words.append(df_list[j]['compound'])
        elif df_list[j]['neg']>0:
            negative_words.append(df_list[j]['compound'])


    number_of_Positive_Words= len(positive_words)
    number_of_Negative_Words= len(negative_words)
    
    
    ratio_pos_sent_to_total_sent=number_of_Positive_Sentences/number_of_total_sentences
    ratio_pos_sent_to_neg_sent=number_of_Positive_Sentences/number_of_Negative_Sentences
    ratio_pos_word_to_total_word=number_of_Positive_Words/ number_of_total_words
    ratio_pos_sent_to_neg_word=number_of_Positive_Words/number_of_Negative_Words

    
    
    book_values = {}
    book_values['Number of positive sentences'] = number_of_Positive_Sentences
    book_values['Number of negative sentences'] = number_of_Negative_Sentences
    book_values['Number of sentences']= number_of_total_sentences
    book_values['Ratio of positive sentence to total sentences'] = ratio_pos_sent_to_total_sent
    book_values['Ratio of positive sentence to negative sentences']= ratio_pos_sent_to_neg_sent
    book_values['Number of positive words'] = number_of_Positive_Words
    book_values['Number of negative words'] = number_of_Negative_Words
    book_values['Number of total words'] = number_of_total_words
    book_values['Ratio of positive words to total words'] = ratio_pos_word_to_total_word
    book_values['Ratio of positive words to negative words']= ratio_pos_sent_to_neg_word
    book_values['Number of words per sentence'] = number_of_words_per_sentences
    book_values['Positivity'] = percentage_pos

    return list(book_values.values())



def scale_data(book_data): 
    import pickle

    scaler = pickle.load(open(scaler_path, "rb")) # load scaler
    data = np.array([book_data]).reshape(1,12) # reshape data for scaler
    scaled_data = scaler.transform(data) # scale data

    return scaled_data


def predict_suitability(scaled_data):
    from tensorflow import keras

    model = keras.models.load_model(model_path) # load model
    suitability = model.predict(scaled_data)[0][0] # predict suitability

    return suitability

