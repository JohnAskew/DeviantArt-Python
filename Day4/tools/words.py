words = ["hello", "water", "hello"]
word_count={}


def most_popular_word(words):
    word_count = {}
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    top_word_count = 0
    top_word = ""
    for word in word_count:
       if word_count[word] > top_word_count:
          top_word_count = word_count[word]
          top_word = word
    print(word_count)
    return top_word
    

most_popular_word(words)
