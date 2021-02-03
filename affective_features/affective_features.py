import spacy
import numpy as np
from py_lex import EmoLex, Liwc
from afinn import Afinn
from senticnet.senticnet import SenticNet
from nltk.corpus import sentiwordnet as swn
import os
import nltk
from nltk.corpus import WordNetCorpusReader
import xml.etree.ElementTree as ET

class Emotion:
    """
    Class Emotion by Clement Michard (c) 2015
    https://github.com/clemtoy/WNAffect
    """

    emotions = {}  # name to emotion (str -> Emotion)

    def __init__(self, name, parent_name=None):
        """Initializes an Emotion object.
            name -- name of the emotion (str)
            parent_name -- name of the parent emotion (str)
        """

        self.name = name
        self.parent = None
        self.level = 0
        self.children = []

        if parent_name:
            self.parent = Emotion.emotions[parent_name] if parent_name else None
            self.parent.children.append(self)
            self.level = self.parent.level + 1

    def get_level(self, level):
        """Returns the parent of self at the given level.
            level -- level in the hierarchy (int)
        """

        em = self
        while em.level > level and em.level >= 0:
            em = em.parent
        return em

    def __str__(self):
        """Returns the emotion string formatted."""

        return self.name

    def nb_children(self):
        """Returns the number of children of the emotion."""

        return sum(child.nb_children() for child in self.children) + 1

    @staticmethod
    def printTree(emotion=None, indent="", last='updown'):
        """Prints the hierarchy of emotions.
            emotion -- root emotion (Emotion)
        """

        if not emotion:
            emotion = Emotion.emotions["root"]

        size_branch = {child: child.nb_children() for child in emotion.children}
        leaves = sorted(emotion.children, key=lambda emotion: emotion.nb_children())
        up, down = [], []
        if leaves:
            while sum(size_branch[e] for e in down) < sum(size_branch[e] for e in leaves):
                down.append(leaves.pop())
            up = leaves

        for leaf in up:
            next_last = 'up' if up.index(leaf) is 0 else ''
            next_indent = '{0}{1}{2}'.format(indent, ' ' if 'up' in last else '│', " " * len(emotion.name))
            Emotion.printTree(leaf, indent=next_indent, last=next_last)
        if last == 'up':
            start_shape = '┌'
        elif last == 'down':
            start_shape = '└'
        elif last == 'updown':
            start_shape = ' '
        else:
            start_shape = '├'
        if up:
            end_shape = '┤'
        elif down:
            end_shape = '┐'
        else:
            end_shape = ''
        print('{0}{1}{2}{3}'.format(indent, start_shape, emotion.name, end_shape))
        for leaf in down:
            next_last = 'down' if down.index(leaf) is len(down) - 1 else ''
            next_indent = '{0}{1}{2}'.format(indent, ' ' if 'down' in last else '│', " " * len(emotion.name))
            Emotion.printTree(leaf, indent=next_indent, last=next_last)

class WNAffect:
    """
    Class WNAffect by Clement Michard (c) 2015
    https://github.com/clemtoy/WNAffect
    """

    def __init__(self, wordnet16_dir, wn_domains_dir):
        """Initializes the WordNet-Affect object."""

        cwd = os.getcwd()
        nltk.data.path.append(cwd)
        wn16_path = "{0}/dict".format(wordnet16_dir)
        self.wn16 = WordNetCorpusReader(os.path.abspath("{0}/{1}".format(cwd, wn16_path)), nltk.data.find(wn16_path))
        self.flat_pos = {'NN': 'NN', 'NNS': 'NN', 'JJ': 'JJ', 'JJR': 'JJ', 'JJS': 'JJ', 'RB': 'RB', 'RBR': 'RB',
                         'RBS': 'RB', 'VB': 'VB', 'VBD': 'VB', 'VGB': 'VB', 'VBN': 'VB', 'VBP': 'VB', 'VBZ': 'VB'}
        self.wn_pos = {'NN': self.wn16.NOUN, 'JJ': self.wn16.ADJ, 'VB': self.wn16.VERB, 'RB': self.wn16.ADV}
        self._load_emotions(wn_domains_dir)
        self.synsets = self._load_synsets(wn_domains_dir)

    def _load_synsets(self, wn_domains_dir):
        """Returns a dictionary POS tag -> synset offset -> emotion (str -> int -> str)."""

        tree = ET.parse("{0}/wn-affect-1.1/a-synsets.xml".format(wn_domains_dir))
        root = tree.getroot()
        pos_map = {"noun": "NN", "adj": "JJ", "verb": "VB", "adv": "RB"}

        synsets = {}
        for pos in ["noun", "adj", "verb", "adv"]:
            tag = pos_map[pos]
            synsets[tag] = {}
            for elem in root.findall(".//{0}-syn-list//{0}-syn".format(pos, pos)):
                offset = int(elem.get("id")[2:])
                if not offset: continue
                if elem.get("categ"):
                    synsets[tag][offset] = Emotion.emotions[elem.get("categ")] if elem.get(
                        "categ") in Emotion.emotions else None
                elif elem.get("noun-id"):
                    synsets[tag][offset] = synsets[pos_map["noun"]][int(elem.get("noun-id")[2:])]

        return synsets

    def _load_emotions(self, wn_domains_dir):
        """Loads the hierarchy of emotions from the WordNet-Affect xml."""

        tree = ET.parse("{0}/wn-affect-1.1/a-hierarchy.xml".format(wn_domains_dir))
        root = tree.getroot()
        for elem in root.findall("categ"):
            name = elem.get("name")
            if name == "root":
                Emotion.emotions["root"] = Emotion("root")
            else:
                Emotion.emotions[name] = Emotion(name, elem.get("isa"))

    def get_emotion(self, word, pos):
        """Returns the emotion of the word.
            word -- the word (str)
            pos -- part-of-speech (str)
        """

        if pos in self.flat_pos:
            pos = self.flat_pos[pos]
            synsets = self.wn16.synsets(word, self.wn_pos[pos])
            if synsets:
                for synset in synsets:
                    offset = synset.offset()
                    if offset in self.synsets[pos]:
                        return self.synsets[pos][offset]
        return None

    def get_emotion_synset(self, offset):
        """Returns the emotion of the synset.
            offset -- synset offset (int)
        """

        for pos in self.flat_pos.values():
            if offset in self.synsets[pos]:
                return self.synsets[pos][offset]
        return None

def wordnetaffect(text, dir_resources):
    list_keys = ['physical-state', 'behaviour', 'trait', 'sensation', 'situation', 'signal', 'cognitive-state', 'cognitive-affective-state', 'mood', 'neutral-emotion', 'ambiguous-emotion', 'positive-emotion', 'negative-emotion']
    list_features = [0] * len(list_keys)
    dict_pos = {'NOUN': 'NN', 'VERB': 'VB', 'ADJ': 'JJ', 'ADV': 'RB'}
    wna = WNAffect(os.path.join(dir_resources,'wordnet-1.6'), os.path.join(dir_resources,'wn-domains-3.2'))

    for token in text:
        if token.pos_ in dict_pos and not token.is_stop:
            emotion = wna.get_emotion(token.lemma_, dict_pos[token.pos_])
            if emotion:
                for i in range(1,5):  # Check the top four levels, excluding 'root'
                    if emotion.get_level(i).name in list_keys:
                        list_features[list_keys.index(emotion.get_level(i).name)] += 1
                        break

    return list_features

def sentiwordnet(text):
    """
    Returns a list with the sum of positive, negative and objective scores for every token in text using SentiWordNet

    :param text: input text pre-processed by Spacy
    :return: a list of three components: positive, negative and objective score
    """
    list_features = [0] * 3
    dict_pos = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}
    for token in text:
        try:
            if token.pos_ in dict_pos and not token.is_stop:
                list_synsets = list(swn.senti_synsets(token.lemma_, dict_pos[token.pos_]))
                if list_synsets:
                    avg_pos = avg_neg = avg_obj = 0
                    for synset in list_synsets:
                        avg_pos += synset.pos_score()
                        avg_neg += synset.neg_score()
                        avg_obj += synset.obj_score()
                    list_features[0] += avg_pos / len(list_synsets)
                    list_features[1] += avg_neg / len(list_synsets)
                    list_features[2] += avg_obj / len(list_synsets)
        except KeyError:
            pass

    return list_features

def senticnet(text):
    """
    Returns a list obtained from SenticNet with the following four features normalized: [pleasantness_value, attention_value, sensiivity_value, aptitude_value]

    :param text: input text pre-processed by Spacy
    :return: a list with the SenticNet features averaged for all the words in text
    """
    list_features = [0] * 4
    sn = SenticNet()
    count_words = 0

    for token in text:
        try:
            concept_info = sn.concept(token)
            list_features[0] += float(concept_info['sentics']['pleasantness'])
            list_features[1] += float(concept_info['sentics']['attention'])
            list_features[2] += float(concept_info['sentics']['sensitivity'])
            list_features[3] += float(concept_info['sentics']['aptitude'])
            count_words += 1
        except KeyError:
            pass

    if count_words != 0:
        list_features = [feature / count_words for feature in list_features]

    return list_features

def dal(text, dir_resources):
    """
    Return a list with the Whisell Dictionary of Affect in Language (DAL) features: pleasantness, activation and imaginery

    :param text: input text pre-processed by Spacy
    :return: a list with the three features computed as the average for each word in text
    """
    list_features = [0] * 3
    with open(os.path.join(dir_resources,'dal.txt')) as file:
        file.readline()  # Skip the header of the dictionary file
        dict_words = dict()
        for line in file:
            list_elements = ' '.join(line.rstrip().split(' ')).split()  # Two splits to remove extra white spaces
            word, pleasantness, activation, imaginery = list_elements
            dict_words[word] = [float(pleasantness), float(activation), float(imaginery)]

        count_words = 0
        for token in text:
            if token in dict_words:
                count_words += 1
                list_features[0] += dict_words[token][0]
                list_features[1] += dict_words[token][1]
                list_features[2] += dict_words[token][2]
        if count_words != 0:
            list_features = [feature/count_words for feature in list_features]

    return list_features

def afinn(text):
    """
    Returns the AFINN sentiment analysis score of an input text, from 6 (very possitive) to -6 (very negative)

    :param text: input text pre-processed by Spacy
    :return: a list with a single element containing the AFINN overall score for the input text
    """
    afinn = Afinn(emoticons = True)

    return [afinn.score(' '.join(text))]

def liwc(text, dir_resources):
    """
    Returns a list with the frequency of 82 aspects contained in the LIWC 2007 dictionary
    Aspects are: ['otherp', 'home', 'period', 'posemo', 'verb', 'discrep', 'present', 'analytic', 'we', 'time', 'death', 'quote', 'auxverb', 'family', 'sexual', 'parenth', 'nonfl', 'percept', 'funct', 'comma', 'feel', 'tentat', 'they', 'achieve', 'quant', 'excl', 'ipron', 'apostro', 'anx', 'pronoun', 'certain', 'relig', 'work', 'semic', 'affect', 'tone', 'cause', 'see', 'health', 'numerals', 'authentic', 'inhib', 'cogmech', 'insight', 'ingest', 'motion', 'sixltr', 'money', 'dash', 'sad', 'you', 'relativ', 'qmark', 'leisure', 'space', 'future', 'colon', 'assent', 'allpct', 'shehe', 'number', 'anger', 'negate', 'past', 'hear', 'bio', 'wc', 'adverb', 'humans', 'preps', 'negemo', 'body', 'article', 'swear', 'social', 'filler', 'exclam', 'ppron', 'i', 'conj', 'friend', 'incl']

    :param text: input text pre-processed by Spacy
    :return: a list o 82 elements with the frequency of each aspect
    """
    lexicon = Liwc(os.path.join(dir_resources,'liwc.dic'))
    list_keys = ['otherp', 'home', 'period', 'posemo', 'verb', 'discrep', 'present', 'analytic', 'we', 'time', 'death', 'quote', 'auxverb', 'family', 'sexual', 'parenth', 'nonfl', 'percept', 'funct', 'comma', 'feel', 'tentat', 'they', 'achieve', 'quant', 'excl', 'ipron', 'apostro', 'anx', 'pronoun', 'certain', 'relig', 'work', 'semic', 'affect', 'tone', 'cause', 'see', 'health', 'numerals', 'authentic', 'inhib', 'cogmech', 'insight', 'ingest', 'motion', 'sixltr', 'money', 'dash', 'sad', 'you', 'relativ', 'qmark', 'leisure', 'space', 'future', 'colon', 'assent', 'allpct', 'shehe', 'number', 'anger', 'negate', 'past', 'hear', 'bio', 'wc', 'adverb', 'humans', 'preps', 'negemo', 'body', 'article', 'swear', 'social', 'filler', 'exclam', 'ppron', 'i', 'conj', 'friend', 'incl']
    list_features = [0] * len(list_keys)  # Initialize to a list of as many zeros as features

    annotation = lexicon.annotate_doc(text)
    for element in annotation:
        for emotion in element:
            index = list_keys.index(emotion)
            list_features[index] += 1

    return list_features

def emolex(text, dir_resources):
    """
    Returns a list with the frequency of 10 emotions in text based on NCR EmoLex lexicon.
    Emotions are: [anger, fear, disgust, negative, surprise, sadness, positive, anticipation, trust, joy]

    :param text: input text pre-processed by Spacy
    :return: a list with the frequncy of each emotion in the input text
    """
    lexicon = EmoLex(os.path.join(dir_resources,'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'))
    list_features = [0] * 10
    list_keys = ['positive', 'negative', 'sadness', 'joy', 'surprise', 'anger', 'disgust', 'trust', 'anticipation', 'fear']

    annotation = lexicon.annotate_doc(text)
    for element in annotation:
        for emotion in element:
            index = list_keys.index(emotion)
            list_features[index] += 1

    return list_features


def extract(nlp, features, line):
    """
    Extract the selected list of features from text and store the result as a NumPy array in file.
    The list of features is a binary list, where 0 indicates to skip the feature and 1 to include it.
    The position of each feature in the list is as follows:
    0: NCR EmoLex features
    1: LIWC 2007 dictionary
    2: AFINN score
    3: Whisell Dictionary of Affect in Language (DAL)
    4: SenticNet
    5: SentiWordNet
    6: WrodNetAffect
    Example: [0,1,0,0,0,1,0] would extract LIWC and SentiWordNet features

    :param nlp: SpaCy pipeline to process text
    :param features: binary list with the features to calculate
    :param line: a line of text with all the information of a Tumblr post from the TSV file
    :param file_name: name of the output file to store the NumPy array (extension .npy is added automatically)
    :param dir_resources: directory where the different dictionaries are stored
    :return: void
    """
    dir_resources = 'affective_features/resources/'
    parsed = nlp(' '.join(line.strip().split()))  # Remove double and trailing white spaces
    text = [token.lemma_ for token in parsed]
    list_features = []

    if features[0]:  # NCR EmoLex features
        list_features += emolex(text, dir_resources)
    if features[1]:  # LIWC 2007 dictionary
        list_features += liwc(text, dir_resources)
    if features[2]:  # AFINN score
        list_features += afinn(text)
    if features[3]:  # Whisell Dictionary of Affect in Language (DAL)
        list_features += dal(text, dir_resources)
    if features[4]:  # SenticNet
        list_features += senticnet(text)
    if features[5]:  # SentiWordNet
        list_features += sentiwordnet(parsed)
    if features[6]:  # WordNetAffect
        list_features += wordnetaffect(parsed, dir_resources)

    #np.save(file_name, list_features)
    return list_features

def start():
    nlp = spacy.load('en_core_web_sm')

    return nlp