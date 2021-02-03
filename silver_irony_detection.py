from numpy.random import seed
seed(100)
from tensorflow import set_random_seed

set_random_seed(2)
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE, RFECV
import seaborn as sns
import matplotlib.pyplot as plt
import keras.callbacks as callbacks
import os
import tensorflow as tf
from keras.applications.vgg16 import decode_predictions
from keras.applications import VGG16
from keras.preprocessing import image as Image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.text import Tokenizer
from attention import attention_3d_block
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input, Multiply, BatchNormalization, LeakyReLU
from keras.layers import Dropout
from keras.regularizers import l1
from keras.regularizers import l2
from keras.models import Model
import pickle
from skimage import feature
# from Bag import BOV
import aidrtokenize as aidrtokenize
import emoji
from tqdm import tqdm
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from scipy.spatial.distance import cosine
from sklearn import svm
from sklearn.utils import shuffle
import performance
from sklearn.naive_bayes import GaussianNB
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import optimizers
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import pandas as pd
import talos
import matplotlib.pyplot as plt

# seed = 2
# np.random.seed(seed)

from keras.layers import Layer
import keras.backend as K

sns.set()


def text_sentiment(text, analyzer):
    vs = analyzer.polarity_scores(text)
    polarity_num = []
    polarity_num.append(float(vs['pos']))
    polarity_num.append(float(vs['compound']))
    polarity_num.append(float(vs['neu']))
    polarity_num.append(float(vs['neg']))
    # print(polarity_num)
    return polarity_num


def embedding_load(embedding_path):
    embedding_vector = {}
    f = open(embedding_path, 'r', encoding='utf8')
    for line in tqdm(f):
        value = line.split(' ')
        word = value[0]
        coef = np.array(value[1:], dtype='float32')
        embedding_vector[word] = coef
    f.close()
    return embedding_vector


def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])

    return clean_text


def text_proprecess(text):
    txt = text.strip().lower()
    txt = give_emoji_free_text(txt)
    txt = aidrtokenize.tokenize(txt)

    return txt


def text_embedding(text, embedding_vector):
    token = Tokenizer()
    token.fit_on_texts(text)
    # print('token')

    vocab_size = len(text.strip().split(' '))
    tags_matrix = []
    zero_array = np.zeros(300)
    for word in text.strip().split(' '):
        if word in embedding_vector.keys():
            tag_embedding = embedding_vector[word]
            tags_matrix.append(np.array(tag_embedding))
            zero_array = zero_array + np.array(tag_embedding)
        else:
            continue
    if len(tags_matrix) > 0:
        tag_feature = zero_array / len(tags_matrix)
    else:
        tag_feature = zero_array

    '''
    embedding_matrix = np.zeros((vocab_size, 300))
    embedding_num = 0
    for word, i in tqdm(token.word_index.items()):
        embedding_value = embedding_vector.get(word)
        if embedding_value is not None:
            embedding_matrix[i] = embedding_value
            embedding_num += 1
    text_embed_feature = embedding_matrix.sum(axis=0)/embedding_num
    '''
    return list(tag_feature)

def dataset_reading(data_path):
    first_images = {}
    with open('data/index_'+data_path.split('.')[0].split('_')[1]+'.txt', 'r', encoding='utf8') as image_file:
        for line in image_file.readlines():
            tumblr_id = line.split(',')[0]  # 138051004348
            # print(tumblr_id)
            tumblr_image = line.split(',')[1].split(' ')[0].replace('\n', '')
            # print(tumblr_image)
            if tumblr_image[-4:] == '.png':
                tumblr_image = tumblr_image.replace('.png', '.jpg')
            elif tumblr_image[-4:] == '.gif':
                tumblr_image = tumblr_image.replace('.gif', '.jpg')
            first_images[tumblr_id] = tumblr_image
    # print(positive_first_image)
    positive_data = pd.read_csv(data_path, sep='\t', header=0, encoding="utf8")  # ISO-8859-1
    ids = positive_data["id"].tolist()
    texts = positive_data["content"].tolist()
    #tags = positive_data["tag"].tolist()
    labels = ['positive'] * len(ids)
    images = []
    for id in ids:
        images.append(first_images[id])
    return ids, texts, images, labels

def tumblr_feature_generation(ids,texts,images,labels,analyzer,data_file_name,label_file_name):
    feature_data_list = []
    label_list = []
    for i in range(len(ids)):
        tumblr_id = ids[i]
        tumblr_text = texts[i]
        tumblr_image = images[i]
        tumblr_label = labels[i]
        try:
            # when using word2vec embedding to represent image tags
            tag_vgg16_feature = im_tag_vgg16_feature_dict[tumblr_image]
            tag_vgg19_feature = im_tag_vgg19_feature_dict[tumblr_image]
            tag_xception_feature = im_tag_xception_feature_dict[tumblr_image]
            tag_inception_feature = im_tag_inception_feature_dict[tumblr_image]
            tag_resnet_feature = im_tag_resnet_feature_dict[tumblr_image]

            # when using bert-like model to represent image tags
            # tag_vgg16_feature = im_tag_vgg16_feature_bertweet_dict[tumblr_image]
            # tag_vgg19_feature = im_tag_vgg19_feature_bertweet_dict[tumblr_image]
            # tag_xception_feature = im_tag_xception_feature_bertweet_dict[tumblr_image]
            # tag_inception_feature = im_tag_inception_feature_bertweet_dict[tumblr_image]
            # tag_resnet_feature = im_tag_resnet_feature_bertweet_dict[tumblr_image]
        except:
            print(tumblr_image)
            continue

        text_bert_feature = text_bert_feature_dict[tumblr_id]
        nlp_feature = text_nlp_feature_dict[tumblr_id]
        affective_feature = text_affective_feature_dict[tumblr_id]

        sentiment_feature, text_embedding_feature = text_feature_extraction(tumblr_text, embedding_vector, analyzer)
        print(text_embedding_feature)
        print('dataset bert feature................', len(text_bert_feature))  # 768
        print('nlp feature ..........', len(nlp_feature))  # 14
        print('affective_feature ..........', len(affective_feature))  # 116
        print(nlp_feature)
        text_feature = text_bert_feature + sentiment_feature + affective_feature + nlp_feature

        tag_embedding_features = np.array(tag_vgg16_feature) + np.array(tag_vgg19_feature) + np.array(
            tag_xception_feature) + np.array(tag_inception_feature) + np.array(tag_resnet_feature)
        tag_embedding_features = tag_embedding_features / 5
        tag_embedding_features = tag_embedding_features.tolist()

        if np.isnan(tag_embedding_features).any():
            tag_embedding_features = [0 for i in range(300)]
        '''
        #when using bert-like model to represent image tags
        tag_embedding_features_bertweet = np.array(tag_vgg16_feature) + np.array(tag_vgg19_feature) + np.array(tag_xception_feature) + np.array(tag_inception_feature) + np.array(tag_resnet_feature)
        tag_embedding_features_bertweet = tag_embedding_features_bertweet / 5
        tag_embedding_features_bertweet = tag_embedding_features_bertweet.tolist()
        '''
        # similarity_feature = np.absolute(np.array(text_embedding_feature) - np.array(tag_embedding_features)).tolist()

        similarity_feature = []
        text_im_similarity_vgg16 = cosine(text_embedding_feature, tag_vgg16_feature)
        text_im_similarity_vgg19 = cosine(text_embedding_feature, tag_vgg19_feature)
        text_im_similarity_resnet = cosine(text_embedding_feature, tag_resnet_feature)
        text_im_similarity_inception = cosine(text_embedding_feature, tag_inception_feature)
        text_im_similarity_xception = cosine(text_embedding_feature, tag_xception_feature)

        if np.isnan(text_im_similarity_vgg16):
            text_im_similarity_vgg16 = 0
        if np.isnan(text_im_similarity_vgg19):
            text_im_similarity_vgg19 = 0
        if np.isnan(text_im_similarity_resnet):
            text_im_similarity_resnet = 0
        if np.isnan(text_im_similarity_inception):
            text_im_similarity_inception = 0
        if np.isnan(text_im_similarity_xception):
            text_im_similarity_xception = 0

        similarity_feature.append(text_im_similarity_vgg16)
        similarity_feature.append(text_im_similarity_vgg19)
        similarity_feature.append(text_im_similarity_resnet)
        similarity_feature.append(text_im_similarity_inception)
        similarity_feature.append(text_im_similarity_xception)

        lbp = im_LBP_feature_dict[tumblr_image]
        other = im_other_feature_dict[tumblr_image]

        tweets_feature = text_feature + lbp + other + tag_embedding_features + similarity_feature
        feature_data_list.append(tweets_feature)
        if np.isnan(tweets_feature).any():
            print(id, tweets_feature)
        label_list.append(tumblr_label)

    np.save(data_file_name, feature_data_list)
    np.save(label_file_name, label_list)

def experimental_data_npy_generation(analyzer):
    positive_ids, positive_texts, positive_images, positive_labels = dataset_reading('data/tu_positive.tsv')
    negative_ids, negative_texts, negative_images, negative_labels = dataset_reading('data/tu_negative.tsv')
    silver_ids = positive_ids+negative_ids
    silver_texts = positive_texts + negative_texts
    silver_images = positive_images + negative_images
    silver_labels = positive_labels + negative_labels
    print('generating train data')
    tumblr_feature_generation(silver_ids,silver_texts,silver_images,silver_labels,analyzer, train_data_file,train_label_file)

def neural_network_model(train_X, train_Y, val_X, val_Y, input_dim):
    callback = callbacks.EarlyStopping(monitor='val_accuracy', patience=100, verbose=1, mode='max')
    tensorboard = TensorBoard(log_dir='log/')
    checkpoint = ModelCheckpoint(filepath='model/best_weights.h5', monitor='val_accuracy', mode='auto',
                                 save_best_only='True')
    callback_lists = [callback, tensorboard, checkpoint]
    # input layer
    input_layer = Input(shape=(input_dim,))
    input_layer1 = BatchNormalization()(input_layer)
    # soft attention
    # attention_probs = Dense(input_dim, activation='softmax', name='attention_vec')(input_layer)
    # multiply
    # attention_mul = Multiply(name='attention_mul')([input_layer, attention_probs])
    # attention_mul = SelfAttLayer()(input_layer)
    # fc
    # dense = Dropout(0.6)(input_layer1)
    dense = Dense(1000)(input_layer1)
    dense = LeakyReLU(alpha=0.3)(dense)
    # dense = Dense(1000)(input_layer1)
    # attention_probs = Dense(1000, activation='softmax', name='attention_vec')(dense)
    # attention_mul = Multiply(name='attention_mul')([dense, attention_probs])
    # dense = BatchNormalization()(dense)
    dense = Dropout(0.6)(dense)
    # fc
    dense = Dense(500)(dense)
    dense = LeakyReLU(alpha=0.3)(dense)
    dense = Dropout(0.6)(dense)
    dense = Dense(50)(dense)
    dense = LeakyReLU(alpha=0.3)(dense)
    # dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)

    # output layer
    output_layer = Dense(1, use_bias=False, activation='sigmoid')(dense)
    model = Model([input_layer], outputs=[output_layer])
    # adam = optimizers.SGD(lr=0.00005, decay=1e-6, momentum=0.9, nesterov=True)
    rmsprop = optimizers.RMSprop(lr=0.0005, rho=0.9, epsilon=1e-06)
    model.compile(optimizer=rmsprop, loss='mse', metrics=["accuracy"])  # adam  rmsprop  logcosh
    history = model.fit(train_X, train_Y, batch_size=64, epochs=100, verbose=1,
                        validation_data=(val_X, val_Y),
                        callbacks=callback_lists, shuffle=True)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("model acc ")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
    return history, model


def text_feature_extraction(text, embedding_vector, analyzer):
    print(text)
    sentiment_feature = text_sentiment(text, analyzer)  # 4
    text = text_proprecess(text)
    # print('sentiment',sentiment_feature)
    # print(text)

    embedding_feature = text_embedding(text, embedding_vector)
    # print('embedding_feature')
    # print(embedding_feature)
    #text_feature = sentiment_feature + list(embedding_feature)
    # print(bert)
    # print(len(bert[0]))
    # text_feature = sentiment_feature + bert[0]
    return sentiment_feature, list(embedding_feature)


def neural_network_classifer(train_X, train_Y, test_X):  #
    n = len(train_X) * 0.9
    X_train = np.array(train_X[:int(n)])  # [validation_size:]   np.array(
    Y_train = np.array(train_Y[:int(n)]).astype('float32')  # [validation_size:]

    X_val = np.array(train_X[int(n):])
    Y_val = np.array(train_Y[int(n):]).astype('float32')

    input_dim = len(train_X[1])
    print(input_dim)
    '''
    p = {'activation': ['relu', 'elu'],
         'optimizer': ['Nadam', 'Adam'],
         'losses': ['binary_crossentropy','logcosh'],
         'batch_size': [16, 32, 64],
         'dropout_rate': [0.2, 0.4, 0.6]}

    def neural_network_model(train_X, train_Y, val_X, val_Y, params):
        callback = callbacks.EarlyStopping(monitor='val_accuracy', patience=100, verbose=1, mode='max')
        tensorboard = TensorBoard(log_dir='log/')
        checkpoint = ModelCheckpoint(filepath='model/best_weights.h5', monitor='val_accuracy', mode='auto',
                                     save_best_only='True')
        callback_lists = [callback, tensorboard, checkpoint]
        # input layer
        input_layer = Input(shape=(input_dim,))
        input_layer1 = BatchNormalization()(input_layer)
        # soft attention
        # attention_probs = Dense(input_dim, activation='softmax', name='attention_vec')(input_layer)
        # multiply
        # attention_mul = Multiply(name='attention_mul')([input_layer, attention_probs])
        # attention_mul = SelfAttLayer()(input_layer)
        # fc
        dense = Dense(1000, activation=params['activation'])(input_layer1)
        # dense = Dense(1000)(input_layer1)
        # attention_probs = Dense(1000, activation='softmax', name='attention_vec')(dense)
        # attention_mul = Multiply(name='attention_mul')([dense, attention_probs])
        # dense = BatchNormalization()(dense)
        dense = Dropout(params['dropout_rate'])(dense)
        # fc
        dense = Dense(500, activation=params['activation'])(dense)
        dense = Dropout(params['dropout_rate'])(dense)
        dense = Dense(100, activation=params['activation'])(dense)
        # dense = BatchNormalization()(dense)
        # dense = Dropout(0.2)(dense)

        # output layer
        output_layer = Dense(1, activation='sigmoid')(dense)
        model = Model([input_layer], outputs=[output_layer])
        adam = optimizers.SGD(lr=0.0006, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=params['optimizer'], loss=params['losses'], metrics=["accuracy"])  # adam  rmsprop
        history = model.fit(train_X, train_Y, batch_size=params['batch_size'], epochs=50, verbose=1,
                            validation_data=(val_X, val_Y),
                            callbacks=callback_lists, shuffle=True)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])

        plt.title("model loss "+params['optimizer']+params['losses']+str(params['batch_size'])+params['activation'])
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(
            "model acc " + params['optimizer'] + params['losses'] + str(params['batch_size']) + params['activation'])
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()
        return history, model

    scan_object = talos.Scan(train_X,train_Y,model=neural_network_model,params=p,experiment_name='irony',fraction_limit=0.5)
    scan_object.details
    '''
    history, model = neural_network_model(X_train, Y_train, X_val, Y_val, input_dim)
    X_test = np.array(test_X)
    # Y_test = np.array(test_Y).astype('float32')

    model.load_weights('model/best_weights.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # loss, acc = model.evaluate(X_test, Y_test, batch_size=64, verbose=0)
    # print(acc)

    result_predict = model.predict(X_test)
    result_predict = [round(list(i)[0]) for i in list(result_predict)]

    return result_predict


if __name__ == '__main__':

    analyzer = SentimentIntensityAnalyzer()

    image_list = []
    text_features = []
    embedding_vector = embedding_load('embedding_300d.txt')  # dict

    train_data_file = 'train_data_npy/silver_affective_nlp_bertweet_train_data_feature.npy'  # _300LBP_300BOW 'mediaeval2015_train_data_feature_vgg.npy'
    train_label_file = 'train_data_npy/silver_affective_nlp_bertweet_train_label_feature.npy'  # 'mediaeval2015_train_label_feature_vgg.npy'
    if os.path.exists(train_data_file):
        train_data = np.load(train_data_file)
        train_data = np.nan_to_num(train_data)
        train_data = train_data.tolist()
        only_text_f = []
        for train in train_data:
            text_feature = train  #
            # print(text_feature)
            only_text_f.append(text_feature)
        train_data = only_text_f
        train_label = np.load(train_label_file).tolist()
        '''
        index_to_delete = []
        for i,e in enumerate(train_label):
            if e == 0:
                index_to_delete.append(i)
        print(index_to_delete[-1100:])
        train_data = [train_data[i] for i in range(0, len(train_data), 1) if i not in index_to_delete[:500]+index_to_delete[-500:]]
        train_label = [train_label[i] for i in range(0, len(train_label), 1) if i not in index_to_delete[:500]+index_to_delete[-500:]]
        '''
        np.random.seed(100)
        data = shuffle(train_data)

        print('共有数据', len(data))
        data = np.array(data)
        # data = train_data
        np.random.seed(100)
        labels = shuffle(train_label)

        print('数据中共有讽刺', labels.count(1))
        print('数据中共有非讽刺', labels.count(0))

        # print(labels)
        # np.random.seed(10)
        # gold_labels = shuffle(gold_label)
        '''
        estimator = []
        estimator.append(('LR', LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200)))
        estimator.append(('SVC', svm.SVC(gamma='auto', probability=True)))
        # estimator.append(('DTC', DecisionTreeClassifier()))
        estimator.append(('RF', RandomForestClassifier(n_estimators=100)))
        # estimator.append(('GNB', GaussianNB()))
        # estimator.append(('XBS', XGBClassifier()))
        estimator.append(('KNN', KNeighborsClassifier(n_neighbors=2)))

        vot_hard = VotingClassifier(estimators=estimator, voting='hard')
        '''
        silver_acc_list = []
        silver_F1_list = []
        silver_P_list = []
        silver_R_list = []

        kf = KFold(n_splits=2)

        for train_index, test_index in kf.split(data):
            # print(train_index)
            X_train = data[train_index]
            test_set = data[test_index]
            Y_train = np.array(labels)[train_index]
            Y_test_set = np.array(labels)[test_index]

            input_dim = len(data[1])
            print(input_dim)

            # vot_hard.fit(X_train, Y_train)
            # silver_results = vot_hard.predict(test_set)
            # gold_results = vot_hard.predict(gold_data)

            silver_results = neural_network_classifer(X_train, Y_train, test_set)

            # clf.fit(X_train, Y_train)
            print('模型训练完成')

            c_silver = confusion_matrix(Y_test_set, silver_results, labels=[0, 1])
            silver_acc, silver_P, silver_R, silver_F1, silver_report = performance.performance_measure(Y_test_set,
                                                                                                       silver_results)
            print('silver_acc', silver_acc)
            print('silver_F1:', silver_F1)
            print('silver_P:', silver_P)
            print('silver_R:', silver_R)
            print(silver_report)
            print(c_silver)
            sns.heatmap(c_silver, annot=True)

            silver_acc_list.append(silver_acc)
            silver_F1_list.append(silver_F1)
            silver_P_list.append(silver_P)
            silver_R_list.append(silver_R)

        print('silver_average_acc', np.mean(silver_acc_list))
        print('silver_average_F1', np.mean(silver_F1_list))
        print('silver_average_P', np.mean(silver_P_list))
        print('silver_average_R', np.mean(silver_R_list))

    else:
        text_bert_feature_dict = np.load('feature_data/silver_bertweet_256_feature.npy').item()
        im_LBP_feature_dict = np.load('feature_data/silver_image_LBP_feature.npy').item()  # 53
        im_tag_inception_feature_dict = np.load('feature_data/silver_image_tag_inception_feature.npy').item()
        im_tag_xception_feature_dict = np.load('feature_data/silver_image_tag_xception_feature.npy').item()
        im_tag_vgg16_feature_dict = np.load('feature_data/silver_image_tag_vgg16_feature.npy').item()
        im_tag_vgg19_feature_dict = np.load('feature_data/silver_image_tag_vgg19_feature.npy').item()
        im_tag_resnet_feature_dict = np.load('feature_data/silver_image_tag_resnet_feature.npy').item()

        #silver_im_tag_bertweet_feature_dict = np.load('feature_data/silver_image_tag_bertweet_feature.npy').item()
        experimental_data_npy_generation(analyzer)


