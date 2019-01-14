#Please import all the packages by pip install

import tweepy
from textblob import TextBlob

#User Authentication
consumer_key = '8x9qorUFn714MYFaflVhtdUAu'
consumer_secret = 'q8NCMHYGBZQX3ZGqO9P1UiRLZpOfq6utkGFuoiph2Lezwnm7FP'

#Access tokens(Tweets)
access_token = '826068351910715392-OjrDZjO8Zg2vfOQz7ODzUQCGOB5HUpS'
access_token_secret = 'rrzR2e9iXhbmkRjxfkqtPLLo0FGBfeKgNSWRDc1vDsmFk'

#Authorization
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
    
#This function fetches tweets based on limit on number of tweets and detects the overall polarity for the given company name which** 
#** will be a mechanism for decision support

def semantic(companyName):
    print("*" *100)
    limit = int(input("\nEnter the limit of tweets you want to make sentitment analysis\n"))
    api = tweepy.API(auth)
    word = companyName
    public_tweets = api.search(word, count=limit)
    positive, null = 0, 0
    for tweet in public_tweets:
        analysis = TextBlob(tweet.text)
        print(tweet.text)
        print(analysis.sentiment)
        print("*" *20)
        
        if analysis.subjectivity ==0:
            null += 1
            next
            
        if analysis.polarity > 0:
            positive += 1

    if positive > ((100-null)/2):
        print("*" *100)
        print(f"The polarity for {companyName} is positive")
        
    else:
        print("*" *100)
        print(f"The polarity for {companyName} is negative")
        
    print("*" *100)
  
    