import spacy
import numpy as np
from collections import Counter
from spacymoji import Emoji

# List of emoticons extracted from SpaCy
emoticons = set(
    r"""
:)
:-)
:))
:-))
:)))
:-)))
(:
(-:
=)
(=
")
:]
:-]
[:
[-:
:o)
(o:
:}
:-}
8)
8-)
(-8
;)
;-)
(;
(-;
:(
:-(
:((
:-((
:(((
:-(((
):
)-:
=(
>:(
:')
:'-)
:'(
:'-(
:/
:-/
=/
=|
:|
:-|
:1
:P
:-P
:p
:-p
:O
:-O
:o
:-o
:0
:-0
:()
>:o
:*
:-*
:3
:-3
=3
:>
:->
:X
:-X
:x
:-x
:D
:-D
;D
;-D
=D
xD
XD
xDD
XDD
8D
8-D
^_^
^__^
^___^
>.<
>.>
<.<
._.
;_;
-_-
-__-
v.v
V.V
v_v
V_V
o_o
o_O
O_o
O_O
0_o
o_0
0_0
o.O
O.o
O.O
o.o
0.0
o.0
0.o
@_@
<3
<33
<333
</3
(^_^)
(-_-)
(._.)
(>_<)
(*_*)
(¬_¬)
ಠ_ಠ
ಠ︵ಠ
(ಠ_ಠ)
¯\(ツ)/¯
(╯°□°）╯︵┻━┻
><(((*>
""".split()
)

def punctuation_count_all(text):
    """
    Frequency of all the punctuation marks in text.

    :param text: text to be processed
    :return: number of punctuation marks found in text
    """
    return len([token for token in text if token.is_punct])

def word_count(text):
    """
    Number of words, discarding punctuation, emoticons and emojis.

    :param text: text to be processed
    :return: number of words in text
    """
    return len([token for token in text if not token.is_punct and not token.text in emoticons and not token._.is_emoji])

def emoticons_count(text):
    """
    Number of emoticons and emojis. Emoticons are ASCII based and emojis are UTF-8 special characters.

    :param text: text to be processed
    :return: total number of emoticons and emojis in text
    """
    total_count = 0
    list_emoticons = []
    for token in text:
        if token.text in emoticons or token._.is_emoji or 'emoji' in token.text:
            total_count += 1
            list_emoticons.append(token)
    return total_count

def pos_count(text):
    """
    Frequency of different part-of-speech tags: verb, noun, adjective and adverb.

    :param text: text to be preocessed
    :return: vector list containing the number of verb, nouns, adjectives and adverbs in text
    """
    dict_pos = Counter(([token.pos_ for token in text]))
    return [dict_pos['VERB'], dict_pos['NOUN'], dict_pos['ADJ'], dict_pos['ADV']]

def character_count(text):
    """
    Total number of characters in text, discarding punctuation, emoticons and emojis.

    :param text: text to be processed
    :return: total number of characters in text
    """
    total_count = 0
    for token in text:
        if not token.is_punct and not token.text in emoticons and not token._.is_emoji:
            total_count += len(token.text)
    return total_count

def punctuation_count(text, mark):
    """
    Frequency of a specific punctuation mark.

    :param text: text to be processed
    :param mark: character representing the punctuation mark to analyze (e.g. ':')
    :return: number of occurrences of mark in text
    """
    return text.count(mark)

def uppercase_count(text):
    """
    Number of uppercase characters in text.

    :param text: text to be processed
    :return: number of uppercase characters
    """
    total_count = 0
    for letter in text:
        if letter.isupper():
            total_count += 1
    return total_count

def pm_count(text):
    """
    Sum of colon, exclamation and question mark frequencies in text.

    :param text: text to be processed
    :return: total number of colon, exclamation and question marks in text
    """
    return punctuation_count(text, ':') + punctuation_count(text, '!') + punctuation_count(text, '?')

def hashtag_count(text):
    """
    Number of hashtags in the post. The hashtags are comma separated values in a specific column of the TSV file.

    :param text: text that contains a list of comma separated values
    :return: number of hashtags in the list provided
    """
    total_count = 0
    if text.strip():
        total_count = len(text.strip().split(','))
    return total_count

def extract(nlp, features, line, tag):
    """
    Extract the selected list of features from text and store the result as a NumPy array in file.
    The list of features is a binary list, where 0 indicates to skip the feature and 1 to include it.
    The position of each feature in the list is as follows:
    0: number of punctuation marks
    1: number of words
    2: number of POS labels
    3: number of characters
    4: number of emoticons and emojis
    5: number of colons
    6: number of exclamations
    7: number of question marks
    8: number of uppercase letters
    9: sum of number of colons, question marks and exclamations
    10: number of hashtags
    Example: [0,1,0,0,0,1,0,0,1,0,0] would extract number of words, number of colons and number of uppercase letters

    :param nlp: SpaCy pipeline to process text
    :param features: binary list with the features to calculate
    :param line: a line of text with all the information of a Tumblr post from the TSV file
    :param file_name: name of the output file to store the NumPy array (extension .npy is added automatically)
    :return: void
    """
    text = nlp(' '.join(line.strip().split()))  # Remove double and trailing white spaces  ' '.join(line.split('\t')[1].strip().split())
    print(text)
    list_features = []

    if features[0]:  # Number of punctuation marks
        list_features.append(punctuation_count_all(text))
    if features[1]:  # Number of words
        list_features.append(word_count(text))
    if features[2]:  # Number of POS labels
        list_features += pos_count(text)
    if features[3]:  # Number of characters
        list_features.append(character_count(text))
    if features[4]:  # Number of emoticons and emojis
        list_features.append(emoticons_count(text))
    if features[5]:  # Number of colons
        list_features.append(punctuation_count(line, ':'))
    if features[6]:  # Number of exclamations
        list_features.append(punctuation_count(line, '!'))
    if features[7]:  # Number of question marks
        list_features.append(punctuation_count(line, '?'))
    if features[8]:  # Number of uppercase letters
        list_features.append(uppercase_count(line))
    if features[9]:  # Sum of number of colons, question marks and exclamations
        list_features.append(pm_count(line))
    if features[10]:  # Number of hashtags
        if tag == '0':
            list_features.append(0)
        else:
            list_features.append(hashtag_count(tag))
    return list_features

    #np.save(file_name, list_features)

def start():
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(Emoji(nlp), first = True)

    return nlp
