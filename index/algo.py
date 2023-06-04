import numpy as np
import os
from django.conf import settings

scaler_path = os.path.join(settings.STATIC_ROOT, 'scaler.pickle')
model_path = os.path.join(settings.STATIC_ROOT, 'model')
all_words_path = os.path.join(settings.STATIC_ROOT, 'all_words.csv')
bad_words_path = os.path.join(settings.STATIC_ROOT, 'bad_words.csv')

def binary_search(arr, low, high, x):
    if high >= low:
        mid = (high+low) // 2
            
        if arr[mid] == x:
            return True
    
        elif arr[mid] > x:
            return binary_search(arr, low, mid - 1, x)
 
        # Else the element can only be present in right subarray
        else:
            return binary_search(arr, mid + 1, high, x)
    else:
        return False


def get_book_data(text, book_location): 

    import pandas as pd

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
    sent_tokens = tokenizer.tokenize(text)
    


    # initialize a lemmatizer and lemmatize all words
    lemmatizer = WordNetLemmatizer()

    lemmatized_words=[]
    for word in filtered_words:
        word2=lemmatizer.lemmatize(word)
        if word!=word2:
            lemmatized_words.append(word2)
        else:
            try:
                lemmatized_words.append(lemmatizer.lemmatize(word, pos="a"))
            except:
                lemmatized_words.append(word2)
                
      
    
    
    all_words = pd.read_csv(all_words_path)
    elem_words_1= all_words["E1"]
    elem_words_2= all_words["E2"]
    elem_words_3= all_words["E3"]
    elem_words_4= all_words["E4"]

    elem_num_1=0
    elem_num_2=0
    elem_num_3=0
    elem_num_4=0
    
    middle_words_1= all_words["M1"]
    middle_words_2= all_words["M2"]
    middle_words_3= all_words["M3"]
    middle_words_4= all_words["M4"]
    middle_words_5= all_words["M5"]

    middle_num_1 = 0
    middle_num_2 = 0
    middle_num_3 = 0
    middle_num_4 = 0
    middle_num_5 = 0

    high_words_1 = all_words["H1"]
    high_words_2 = all_words["H2"]
    high_words_3 = all_words["H3"]
    high_words_4 = all_words["H4"]
    high_words_5 = all_words["H5"]
    high_words_6 = all_words["H6"]

    high_num_1 = 0
    high_num_2 = 0
    high_num_3 = 0
    high_num_4 = 0
    high_num_5 = 0
    high_num_6 = 0

    
    for word in lemmatized_words:
        length=100
        result = binary_search(elem_words_1, 0, length-1, word)
        if (result):
            elem_num_1=elem_num_1+1
            
        length=174
        result2 = binary_search(elem_words_2, 0, length-1, word)
        if (result2):
            elem_num_2=elem_num_2+1
        
        length=223
        result3 = binary_search(elem_words_3, 0, length-1, word)
        if (result3):
            elem_num_3=elem_num_3+1
            
        length=95
        result4 = binary_search(elem_words_4, 0, length-1, word)
        if (result4):
            elem_num_4=elem_num_4+1
            
        length=105
        result5 = binary_search(middle_words_1, 0, length-1, word)
        if (result5):
            middle_num_1=middle_num_1+1
            
        length=299
        result6 = binary_search(middle_words_2, 0, length-1, word)
        if (result6):
            middle_num_2=middle_num_2+1
        
        length=400
        result7 = binary_search(middle_words_3, 0, length-1, word)
        if (result7):
            middle_num_3=middle_num_3+1
            
        length=499
        result8 = binary_search(middle_words_4, 0, length-1, word)
        if (result8):
            middle_num_4=middle_num_4+1
            
        length=327
        result9 = binary_search(middle_words_5, 0, length-1, word)
        if (result9):
            middle_num_5=middle_num_5+1
        
        length=326
        result10 = binary_search(high_words_1, 0, length-1, word)
        if (result10):
            high_num_1=high_num_1+1
            
        length=417
        result11 = binary_search(high_words_2, 0, length-1, word)
        if (result11):
            high_num_2=high_num_2+1
        
        length=495
        result12 = binary_search(high_words_3, 0, length-1, word)
        if (result12):
            high_num_3=high_num_3+1
            
        length=395
        result13 = binary_search(high_words_4, 0, length-1, word)
        if (result13):
            high_num_4=high_num_4+1
            
        length=224
        result14 = binary_search(high_words_5, 0, length-1, word)
        if (result14):
            high_num_5=high_num_5+1
            
        length=158
        result15 = binary_search(high_words_6, 0, length-1, word)
        if (result15):
            high_num_5=high_num_5+1

    
    num_elem_words = elem_num_1+elem_num_2+elem_num_3+elem_num_4
    num_middle_words = middle_num_1+middle_num_2+middle_num_3+middle_num_4+middle_num_5
    num_high_words =  high_num_1+high_num_2+high_num_3+high_num_4+high_num_5+high_num_6
    
    elem_words_per=num_elem_words/len(lemmatized_words)
    middle_words_per=num_middle_words/len(lemmatized_words)
    high_words_per=num_high_words/len(lemmatized_words)

    #VADER            
    sia = SentimentIntensityAnalyzer()
    df_list = []
    df_sent_list=[]

    with open(bad_words_path, "r") as f: 
        bad_words = [line.replace("\n", "") for line in f.readlines()]
    
    
    for word in lemmatized_words: 
        word_info = {}
        word_info["word"] = word
        result = binary_search(bad_words, 0, len(bad_words)-1, word)
        if (result):
            word_info.update({"neg": 1.0,"neu": 0.0, "pos": 0.0, "compound": -1.0})
        else: 
            word_info.update(sia.polarity_scores(word))
        df_list.append(word_info)
    
    
    bad_words_long = []
    for word in bad_words:
        index = word.find(" ")
        if index >= 0:
            bad_words_long.append(word)
            bad_words.remove(word)

    for sentence in sent_tokens:
        sent_info={}
        sent_info["sentence"]=sentence
        words = sentence.split()
        for word in words:
            result = binary_search(bad_words, 0, len(bad_words)-1, word)
            if (result):
                sent_info.update({"neg": 1.0,"neu": 0.0, "pos": 0.0, "compound": -1.0})
                break
            else:
                '''
                for bword in bad_words_long:
                    if (sentence.find(bword)>=0):
                        sent_info.update({"neg": 1.0,"neu": 0.0, "pos": 0.0, "compound": -1.0})
                    else:
                '''
                sent_info.update(sia.polarity_scores(sentence))
        df_sent_list.append(sent_info)
    


    positive_sentence=[]
    negative_sentence=[]
    
    positive_sentence_sum = []
    negative_sentence_sum = []
    
    for b in range(len(df_sent_list)):
        if df_sent_list[b]['compound'] > 0:
            positive_sentence.append(df_sent_list[b]['compound'])
        elif df_sent_list[b]['neg']>0:
            negative_sentence.append(df_sent_list[b]['compound'])

    for item in positive_sentence:
        positive_sentence_sum.append(np.power(item, 14))
        
    for item in negative_sentence:
        negative_sentence_sum.append(np.power(item, 14))

    sum_neg = np.sum(negative_sentence_sum) 
    sum_pos = np.sum(positive_sentence_sum)  

    total_sum = sum_neg + sum_pos

    percentage_neg = (sum_neg/ total_sum)*100
    percentage_pos = (sum_pos/ total_sum)*100

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
    book_values['Ratio of Middle School Words to Total Words']=middle_words_per
    book_values['Ratio of High School Words to Total Words']=high_words_per
    book_values['Positivity'] = percentage_pos

    return list(book_values.values())

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
    data = np.array([book_data]).reshape(1,14) # reshape data for scaler
    scaled_data = scaler.transform(data) # scale data

    return scaled_data


def predict_suitability(scaled_data):
    from tensorflow import keras

    model = keras.models.load_model(model_path) # load model
    suitability = model.predict(scaled_data)[0][0] # predict suitability

    return suitability

