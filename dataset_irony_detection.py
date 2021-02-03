from numpy.random import seed
seed(100)
from tensorflow import set_random_seed
set_random_seed(2)
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import KFold
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
from keras.models import Model
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
#from xgboost import XGBClassifier
#import talos
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
    return list(tag_feature)

def dataset_reading(data_path):
    ids = []
    labels = []
    images = []
    texts = []
    with open(data_path, 'r', encoding='utf8') as train_file:
        for line in train_file.readlines():
            print(line)
            tumblr = line.replace('\n', '')[1:-1]  # 138051004348
            tumblr_id = tumblr[1:19]
            # print(tumblr_id)
            tumblr_image = str(tumblr_id) + '.jpg'
            tumblr_text = str(tumblr[23:-4])
            print(tumblr_text)
            tumblr_label = int(tumblr[-1])
            print(tumblr_label)
            ids.append(tumblr_id)
            labels.append(tumblr_label)
            images.append(tumblr_image)
            texts.append(tumblr_text)
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
        david_feature = text_nlp_feature_dict[tumblr_id]
        affective_feature = text_affective_feature_dict[tumblr_id]

        sentiment_feature, text_embedding_feature = text_feature_extraction(tumblr_text, embedding_vector, analyzer)
        print(text_embedding_feature)
        print('dataset bert feature................', len(text_bert_feature))  # 768
        print('david feature ..........', len(david_feature))  # 14
        print('affective_feature ..........', len(affective_feature))  # 116
        print(david_feature)
        text_feature = text_bert_feature + sentiment_feature + affective_feature + david_feature

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
    train_ids, train_texts, train_images, train_labels = dataset_reading('data/dataset_texts/train.txt')
    val_ids, val_texts, val_images,val_labels = dataset_reading('data/dataset_texts/valid2.txt')
    test_ids, test_texts, test_images, test_labels = dataset_reading('data/dataset_texts/test2.txt')
    print('generating train data')
    tumblr_feature_generation(train_ids,train_texts,train_images,train_labels,analyzer, train_data_file,train_label_file)
    print('generating val data')
    tumblr_feature_generation(val_ids, val_texts, val_images, val_labels, analyzer, val_data_file, val_label_file)
    print('generating test data')
    tumblr_feature_generation(test_ids, test_texts, test_images, test_labels, analyzer, test_data_file, test_label_file)

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
    #dense = Dropout(0.1)(input_layer1)
    dense = Dense(1000)(input_layer1)  #, activation='relu'
    dense = LeakyReLU(alpha=0.3)(dense)
    # dense = Dense(1000)(input_layer1)
    # attention_probs = Dense(1000, activation='softmax', name='attention_vec')(dense)
    # attention_mul = Multiply(name='attention_mul')([dense, attention_probs])
    # dense = BatchNormalization()(dense)
    dense = Dropout(0.6)(dense)
    # fc
    dense = Dense(500)(dense)  #glorot_uniform  , activation='linear'  , init='glorot_uniform'
    dense = LeakyReLU(alpha=0.3)(dense)
    dense = Dropout(0.6)(dense)
    dense = Dense(50)(dense)
    dense = LeakyReLU(alpha=0.3)(dense)
    # dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)

    # output layer
    output_layer = Dense(1, use_bias=False, activation='sigmoid')(dense)  #
    model = Model([input_layer], outputs=[output_layer])
    #sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    rmsprop = optimizers.RMSprop(lr=0.0005, rho=0.9, epsilon=1e-06)

    #adam = optimizers.Adam(lr=0.0001, decay=1e-6,)
    model.compile(optimizer=rmsprop, loss='mse', metrics=["accuracy"])  # adam  rmsprop  logcosh  mse
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
    text = text_proprecess(text)
    # print('sentiment',sentiment_feature)
    # print(text)
    embedding_feature = text_embedding(text, embedding_vector)
    sentiment_feature = text_sentiment(text, analyzer)
    return sentiment_feature, list(embedding_feature)

def neural_network_classifer(train_X, train_Y, val_data, val_label, test_data):  # test_X

    X_train = np.array(train_X)  # [validation_size:]   np.array(
    Y_train = np.array(train_Y).astype('float32')  # [validation_size:]

    X_val = np.array(val_data)
    Y_val = np.array(val_label).astype('float32')

    X_test = np.array(test_data)

    input_dim = len(train_X[1])
    print('feature length ',input_dim)
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
    # X_test = np.array(test_X)
    # Y_test = np.array(test_Y).astype('float32')

    model.load_weights('model/best_weights.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # loss, acc = model.evaluate(X_test, Y_test, batch_size=64, verbose=0)
    # print(acc)

    # result_predict = model.predict(X_test)
    # result_predict = [round(list(i)[0]) for i in list(result_predict)]

    test_predict = model.predict(X_test)
    test_predict = [round(list(i)[0]) for i in list(test_predict)]

    return test_predict


if __name__ == '__main__':

    analyzer = SentimentIntensityAnalyzer()

    image_list = []
    text_features = []
    embedding_vector = embedding_load('embedding_300d.txt')

    dataset_name = 'dataset'

    train_data_file = 'train_data_npy/dataset_sentiment_affective_nlp_bertweet_image_train_data_feature1.npy'
    train_label_file = 'train_data_npy/dataset_sentiment_affective_nlp_bertweet_image_train_label_feature1.npy'
    val_data_file = 'train_data_npy/dataset_sentiment_affective_nlp_bertweet_image_val_data_feature1.npy'
    val_label_file = 'train_data_npy/dataset_sentiment_affective_nlp_bertweet_image_val_label_feature1.npy'
    test_data_file = 'train_data_npy/dataset_sentiment_affective_nlp_bertweet_image_test_data_feature1.npy'
    test_label_file = 'train_data_npy/dataset_sentiment_affective_nlp_bertweet_image_test_label_feature1.npy'

    if os.path.exists(train_data_file):
        train_data = np.load(train_data_file)
        train_data = np.nan_to_num(train_data)
        train_data = train_data.tolist()
        only_text_f = []
        for train in train_data:
            text_feature = train  #[:768]+train[772:]
            only_text_f.append(text_feature)
        train_data = only_text_f
        train_label = np.load(train_label_file).tolist()

        val_data = np.load(val_data_file)
        val_data = np.nan_to_num(val_data)
        val_data = val_data.tolist()
        only_val_f=[]
        for val in val_data:
            text_feature = val  #[:768]+val[772:]
            # print(text_feature)
            only_val_f.append(text_feature)
        val_data = only_val_f
        val_label = np.load(val_label_file).tolist()

        test_data = np.load(test_data_file)
        test_data = np.nan_to_num(test_data)
        test_data = test_data.tolist()
        only_test_f = []
        for test in test_data:
            text_feature = test  #[:768]+test[772:]
            only_test_f.append(text_feature)
        test_data = only_test_f
        test_label = np.load(test_label_file).tolist()

        np.random.seed(100)
        data = shuffle(train_data)

        print('data count', len(data))
        data = np.array(data)
        # data = train_data
        np.random.seed(100)
        labels = shuffle(train_label)

        print('irony data number', labels.count(1))
        print('non-irony data number', labels.count(0))

        val_labels = np.array(val_label)
        test_labels = np.array(test_label)

        val_data = np.array(val_data)
        test_data = np.array(test_data)
        '''
        estimator = []
        estimator.append(('LR', LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200)))
        estimator.append(('SVC', svm.SVC(gamma='auto', probability=True)))
        #estimator.append(('DTC', DecisionTreeClassifier()))
        estimator.append(('RF', RandomForestClassifier(n_estimators=100)))
        estimator.append(('GNB', GaussianNB()))
        estimator.append(('XBS', XGBClassifier()))
        estimator.append(('KNN', KNeighborsClassifier(n_neighbors=2)))

        vot_hard = VotingClassifier(estimators=estimator, voting='hard')


        input_dim = len(train_data[1])
        print(input_dim)

        #vot_hard.fit(data, labels)
        #test_results = vot_hard.predict(test_data)
        #silver_results, gold_50_results, gold_80_results, gold_100_results = neural_network_classifer(X_train, Y_train, test_set, gold_50_data, gold_80_data, gold_100_data)
        #clf = GaussianNB()
        clf = LogisticRegression() #RandomForestClassifier(n_estimators=100)
        #clf = svm.SVC()
        #clf = XGBClassifier()
        #clf = LogisticRegression(random_state=0)

        clf.fit(data, labels)
        print('模型训练完成')
        test_results = clf.predict(test_data)
        #gold_results = clf.predict(gold_data)
        c_silver = confusion_matrix(test_labels, test_results, labels=[0, 1])
        silver_acc,silver_P,silver_R,silver_F1,silver_report = performance.performance_measure(test_labels, test_results,)
        print('silver_acc',silver_acc)
        print('silver_F1:',silver_F1)
        print('silver_P:', silver_P)
        print('silver_R:', silver_R)
        print(silver_report)
        print(c_silver)
        importance = clf.coef_
        print(importance)
        for i,v in enumerate(importance[0]):
            print('feature: %0d, score: %.5f' % (i,v))


        '''
        test_results = neural_network_classifer(train_data, train_label, val_data, val_label, test_data)
        c50 = confusion_matrix(test_labels, test_results, labels=[0, 1])
        test_acc, test_P, test_R, test_F1, test_report = performance.performance_measure(test_labels, test_results)
        print('acc', test_acc)
        print('F1:', test_F1)
        print('P:', test_P)
        print('R:', test_R)
        print(test_report)
        print(c50)
    else:
        print('there are no training data')
        # feature data loading
        text_bert_feature_dict = np.load('feature_data/' + dataset_name + '_bertweet_256_feature.npy').item()  # 768 0r 1000
        text_affective_feature_dict = np.load('feature_data/' + dataset_name + '_affective_feature.npy').item()
        text_nlp_feature_dict = np.load('feature_data/' + dataset_name + '_nlp_feature.npy').item()

        im_LBP_feature_dict = np.load('feature_data/' + dataset_name + '_images_image_LBP_feature.npy').item()  # 59
        im_other_feature_dict = np.load('feature_data/' + dataset_name + '_images_image_other_feature.npy').item()  # 7

        # tag representation by word2vec embedding
        im_tag_inception_feature_dict = np.load('feature_data/' + dataset_name + '_image_tag_inception_feature.npy').item()
        im_tag_xception_feature_dict = np.load('feature_data/' + dataset_name + '_image_tag_xception_feature.npy').item()
        im_tag_vgg16_feature_dict = np.load('feature_data/' + dataset_name + '_image_tag_vgg16_feature.npy').item()
        im_tag_vgg19_feature_dict = np.load('feature_data/' + dataset_name + '_image_tag_vgg19_feature.npy').item()
        im_tag_resnet_feature_dict = np.load('feature_data/' + dataset_name + '_image_tag_resnet_feature.npy').item()

        # tag representation by bert-like model
        # im_tag_inception_feature_bertweet_dict = np.load('feature_data/'+dataset_name+'_inception_bertweet_feature.npy').item()
        # im_tag_xception_feature_bertweet_dict = np.load('feature_data/'+dataset_name+'_xception_bertweet_feature.npy').item()
        # im_tag_vgg16_feature_bertweet_dict = np.load('feature_data/'+dataset_name+'_vgg16_bertweet_feature.npy').item()
        # im_tag_vgg19_feature_bertweet_dict = np.load('feature_data/'+dataset_name+'_vgg19_bertweet_feature.npy').item()
        # im_tag_resnet_feature_bertweet_dict = np.load('feature_data/'+dataset_name+'_resnet_bertweet_feature.npy').item()

        experimental_data_npy_generation(analyzer)





