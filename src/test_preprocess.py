from preprocess import clean_tweet

tweet = "I absolutely LOVE this airline!!! 😍😍 Visit https://test.com #Amazing @VirginAmerica"

print("Original Tweet:")
print(tweet)

print("\nCleaned Tweet:")
print(clean_tweet(tweet))