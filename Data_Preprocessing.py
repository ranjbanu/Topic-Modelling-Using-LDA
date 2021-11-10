# install the necessary libraries using 
# !pip install texthero 

#import libraries

import numpy as np
import pandas as pd
import spacy
import math
import re
import nltk
import texthero as hero
from spacy.language import Language
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize

# read the .csv file
df = pd.read_csv("Cleaned_cars.csv")

# clean the text using texthero library removing the punctuations
df['cleaned_complaint_body'] = hero.remove_brackets(df['new_complaint_body'])
df['cleaned_complaint_body']=hero.remove_whitespace(df['cleaned_complaint_body'])
df['cleaned_complaint_body'] = hero.remove_square_brackets(df['cleaned_complaint_body'])
df['cleaned_complaint_body'] = hero.remove_punctuation(df['cleaned_complaint_body'])

# function to clean the text from contractions, paragraph numbers, new line characters and finally convert the text into lower case
def clean_data(text):
    text = str(text).lower()
    # removing links
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    # removing the paragraph numbers
    text = re.sub('[0-9]+.\t','',text)
    # removing new line characters
    text = re.sub('\n ','',text)
    text = re.sub('\n',' ',text)
  # Contractions
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"We're", "We are", text)
    text = re.sub(r"That's", "That is", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"they're", "they are", text)
    text = re.sub(r"Can't", "Cannot", text)
    text = re.sub(r"wasn't", "was not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"aren't", "are not", text)
    text = re.sub(r"isn't", "is not", text)
    text = re.sub(r"What's", "What is", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"hasn't", "has not", text)
    text = re.sub(r"There's", "There is", text)
    text = re.sub(r"He's", "He is", text)
    text = re.sub(r"It's", "It is", text)
    text = re.sub(r"You're", "You are", text)
    text = re.sub(r"I'M", "I am", text)
    text = re.sub(r"shouldn't", "should not", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"i'm", "I am", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r"Isn't", "is not", text)
    text = re.sub(r"Here's", "Here is", text)
    text = re.sub(r"you've", "you have", text)
    text = re.sub(r"you've", "you have", text)
    text = re.sub(r"we're", "we are", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"we've", "we have", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"It's", "It is", text)
    text = re.sub(r"Here's", "Here is", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"I've", "I have", text)
    text = re.sub(r"y'all", "you all", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"would've", "would have", text)
    text = re.sub(r"it'll", "it will", text)
    text = re.sub(r"we'll", "we will", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"We've", "We have", text)
    text = re.sub(r"he'll", "he will", text)
    text = re.sub(r"Y'all", "You all", text)
    text = re.sub(r"Weren't", "Were not", text)
    text = re.sub(r"Didn't", "Did not", text)
    text = re.sub(r"they'll", "they will", text)
    text = re.sub(r"they'd", "they would", text)
    text = re.sub(r"DON'T", "DO NOT", text)
    text = re.sub(r"That's", "That is", text)
    text = re.sub(r"they've", "they have", text)
    text = re.sub(r"i'd", "I would", text)
    text = re.sub(r"should've", "should have", text)
    text = re.sub(r"You're", "You are", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"Don't", "Do not", text)
    text = re.sub(r"we'd", "we would", text)
    text = re.sub(r"i'll", "I will", text)
    text = re.sub(r"weren't", "were not", text)
    text = re.sub(r"They're", "They are", text)
    text = re.sub(r"Can't", "Cannot", text)
    text = re.sub(r"you'll", "you will", text)
    text = re.sub(r"I'd", "I would", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"you're", "you are", text)
    text = re.sub(r"i've", "I have", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"i'll", "I will", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"i'd", "I would", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"ain't", "am not", text)
    text = re.sub(r"you'll", "you will", text)
    text = re.sub(r"I've", "I have", text)
    text = re.sub(r"Don't", "do not", text)
    text = re.sub(r"I'll", "I will", text)
    text = re.sub(r"I'd", "I would", text)
    text = re.sub(r"Let's", "Let us", text)
    text = re.sub(r"you'd", "You would", text)
    text = re.sub(r"It's", "It is", text)
    text = re.sub(r"Ain't", "am not", text)
    text = re.sub(r"Haven't", "Have not", text)
    text = re.sub(r"Could've", "Could have", text)
    text = re.sub(r"you've", "you have", text)  
    text = re.sub(r"don't", "do not", text)    
    text = text.strip()
    return text
    
  df['cleaned_complaint_body'] = df['cleaned_complaint_body'].apply(clean_data)

  # function to remove customized stop words and stop words in English
  def spacy_process(text):
      custom_stop_words = ['I','sir/','mr.','mrs.','ms.','dear','sir','mam','madam','medam','medem','madame','hello',
                           'respected','team','everyone','protected','main', 'nhi', 'car', 'center', 'maine', 'meri', 
                           'tha', 'diya', 'nahi', 'aur', 'mene', 'gai', 'hai', 'liye', 'bhi', 'thi', 'gadi', 'kya', 'tak', 
                           'koi', 'mujhe', 'laga', 'naam', 'kar', 'kuch','please','plz','pls','sai','mandovi','protected',
                           'protect','protecte','servicing','husband','wife','hi','ki', 'se' ,'ko' 'ke', 'ka', 'ho' ,
                           'mai', 'par' ,'gaya']
      doc = nlp(text)  
      lemma_list = []
      for token in doc:
          lemma_list.append(token.lemma_)
      filtered_wordlist =[] 
      spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

      for word in lemma_list:
          if word not in spacy_stopwords:
              if word not in custom_stop_words:
                  filtered_wordlist.append(word) 
      punctuations="?:!.,;&"
      for word in filtered_wordlist:
          if word in punctuations:
              filtered_wordlist.remove(word)

      return ' '.join(filtered_wordlist)
    
    df['cleaned_complaint_body'] = df['cleaned_complaint_body'].apply(spacy_process)  
    
    # function to obtain the car model and brand
    def identify_car_model(text,car):
      car_model =""
      brand = ""
      lsttext = list(text.split())
     # maruti_car_models = ["swift","baleno","ertiga","alto","vitara brezza","dzire","wagon r","celerio",'ritz'
                      #     ,'wagonr','ciaz','scooter','k10','gixxer','access','burgman','gsx']
      car_models = {'baleno':'maruti-suzuki','swift':'maruti-suzuki','dzire':'maruti-suzuki','brezza':'maruti-suzuki',
                    'wagon r':'maruti-suzuki','wagonr':'maruti-suzuki','ertiga':'maruti-suzuki','celerio':'maruti-suzuki',
                    's-presso':'maruti-suzuki','alto':'maruti-suzuki','ciaz':'maruti-suzuki','ignis':'maruti-suzuki',
                    'eeco':'maruti-suzuki','vitara brezza':'maruti-suzuki','alto 800':'maruti-suzuki',

                    'venue':'hyundai','creta':'hyundai','i20':'hyundai','verna':'hyundai','i10':'hyundai','santro':'hyundai','aura':'hyundai',
                    'alcazar':'hyundai','elantra':'hyundai',

                    'bolero':'mahindra','bolero bs6':'mahindra','bolerobs6':'mahindra','tuv300':'mahindra','bulero':'mahindra','thaar':'mahindra',
                    'xuv 300':'mahindra','thar':'mahindra','xuv w6':'mahindra','xuv300':'mahindra','xuvw6':'mahindra','kuv100nxt':'mahindra',
                    'xuv 500':'mahindra','xuv500':'mahindra','scorpio':'mahindra','alturas g4':'mahindra','kuv100k2':'mahindra','tuv 300':'mahindra',
                    'marazzo':'mahindra','alturasg4':'mahindra','xuv3oo w4':'mahindra','kuv 100 k2':'mahindra','kuv 100 nxt':'mahindra',
                    'kuv100':'mahindra','marazo':'mahindra','morazzo':'mahindra','mahendra':'mahindra','mahindra':'mahindra','xuv5oo':'mahindra',

                    'polo':'volkswagen','vento':'volkswagen','taigun':'volkswagen','t-roc':'volkswagen',

                    'ecosport':'ford','endeavour':'ford','endeavor':'ford','figo':'ford','aspire':'ford','freestyle':'ford',
                    'fiesta':'ford','eco sport':'ford','classic':'ford','figi':'ford','ikon':'ford','escort':'ford','ford':'ford',

                    'duster':'renault','kwid':'renault','triber':'renault','kiger':'renault','logan':'renault',

                    'fortuner':'toyota','innova':'toyota','innovacrystal':'toyota','camry':'toyota','yaris':'toyota','glanza':'toyota',
                    'kushaq':'skoda','octavia':'skoda','superb':'skoda','rapid':'skoda',

                    'magnite':'nissan','kicks':'nissan','gt-r':'nissan',

                    'aveo':'chevrolet','beat':'chevrolet','captiva':'chevrolet','corvette':'chevrolet','cruze':'chevrolet','enjoy':'chevrolet','forester':'chevrolet','optra':'chevrolet','sail':'chevrolet',
                    'spark':'chevrolet','tavera':'chevrolet','trailblazer':'chevrolet','camaro':'chevrolet',

                    'abarth':'fiat','adventure':'fiat','punto':'fiat','linea':'fiat','palio':'fiat','petra':'fiat',
                    'siena':'fiat','uno':'fiat',

                    'q2':'audi','e-tron':'audi','a6':'audi','a4':'audi','q8':'audi','audi6':'audi','q3':'audi',

                    'gla':'mercedes','gls':'mercedes','gle':'mercedes',

                    'goplus':'datsun','redigo':'datsun','go':'datsun',

                    'tigor ev':'tata motors','tigorev':'tata motors','nexon':'tata motors','harrier':'tata motors','safari':'tata motors','tiago':'tata motors'
                    ,'tiago nrg':'tata motors','tiagonrg':'tata motors','altroz':'tata motors','tata':'tata motors',
                    'nexonev':'tata motors','punch':'tata motors','indica':'tata motors','sierra':'tata motors','nano':'tata motors','indica':'tata motors'
                    ,'indigo':'tata motors','manza':'tata motors','maanza':'tata motors','indica vista':'tata motors','hexa':'tata motors','storme':'tata motors'
                   }
    for model in car_models:
        part_txt = list(model.split())
        #print(part_txt[0])
        if part_txt[0] in lsttext: #re.search(str(model),text):
            car_model = model
            if car =='brand':
                brand = (car_models[car_model])
            elif car =='model':
                brand = part_txt[0]
            break
    return brand
    
    df['Brand']= df['cleaned_complaint_body'].apply(identify_car_model,car='brand')
    df['car_model']= df['cleaned_complaint_body'].apply(identify_car_model,car='model')

    # Analysis the negativity of the sentiment using the SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    df['sentiment_scores']=df['cleaned_complaint_body'].apply(lambda txt: sid.polarity_scores(txt))
    df['final_sentiment_score']=df['sentiment_scores'].apply(lambda d:d['compound'])

    # Rating the severity of issues depending on the score of the final sentiment score
    df['complaint_severity']=df['final_sentiment_score'].apply(lambda score: 'low' if score >=0.90 
                                                                                   else 'moderate' if (score >= 0.75 and score< 0.90)
                                                                                   else 'high' if (score >= 0.50 and score< 0.75)
                                                                                   else 'very high' if (score >= 0.25 and score< 0.50)
                                                                                   else 'urgent'  if (score >= 0.05 and score< 0.25)
                                                                                   else 'very urgent' if (score >= -0.05 and score< 0.05)
                                                                                   else 'top most' )

    df.head()
    df.to_csv('Cleaned_cars.csv',index=False)                                                                          

