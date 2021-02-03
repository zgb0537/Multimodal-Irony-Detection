import cv2,os,PIL
import numpy as np
from keras.applications.vgg16 import decode_predictions
from keras.applications import ResNet50, Xception, InceptionV3, VGG16, VGG19
from keras.preprocessing import image as Image
from keras.applications.vgg16 import preprocess_input
from tqdm import tqdm
from skimage import feature
from keras.models import Model
import math
import pandas as pd
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import whiten

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
            self.radius, method="nri_uniform")
        #print(list(lbp.ravel()))
        #print(set(list(lbp.ravel())))
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints*(self.numPoints-1) + 3),
            range=(0, self.numPoints*(self.numPoints-1) + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist

def image_feature_extraction(x, model):
    data = preprocess_input(x)
    layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    features = layer_model.predict(data)
    print(im)
    print(features.shape)
    # print(features)
    return features

def image_tags_extraction(x, model):
    data = preprocess_input(x)
    yhat = model.predict(data)
    labels = decode_predictions(yhat, top=10)[0]
    print(labels)
    return labels

def image_colorfulness(image):
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(image.astype("float"))
    # compute rg = R - G
    rg = np.absolute(R - G)
    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)


def im_tags_embedding(labels, embedding_vector):
    #print(labels[0])
    words = []
    for label in labels:
        word = label[1]
        #print(word)
        words.append(word)

    tags_matrix = []
    zero_array = np.zeros(300)
    for tag in words:
        if tag in embedding_vector.keys():
            tag_embedding = embedding_vector[tag]
            tags_matrix.append(np.array(tag_embedding))
            zero_array = zero_array+np.array(tag_embedding)
    tag_feature = zero_array / len(tags_matrix)
    return list(tag_feature)


def im_color_hist(im):

    chans = cv2.split(im)
    colors = ("b", "g", "r")
    features = []
    for (chan, color) in zip(chans, colors):
        average_value = chan.mean()/256
        features.append(average_value.item())
        '''
        hist = cv2.calcHist([chan],[0], None, [256], [0,255])
        np.array(hist).flatten()
        hist.resize(hist.size)
        hist = list(hist/np.sum(hist))
        features.extend(hist)
        '''
    return features

def dominant_color_rgb(image):
    r = []
    g = []
    b = []
    for line in image:
        for pixel in line:
            temp_r, temp_g, temp_b = pixel
            r.append(temp_r)
            g.append(temp_g)
            b.append(temp_b)
    df = pd.DataFrame({'red':r,'blue':b,'green':g})
    df['scaled_red'] = whiten(df['red'])
    df['scaled_blue'] = whiten(df['blue'])
    df['scaled_green'] = whiten(df['green'])

    cluster_center, distortion = kmeans(df[['scaled_red','scaled_green','scaled_blue']],1)
    #print(cluster_center)
    return cluster_center

def embedding_load(embedding_path):
    embedding_vector = {}
    f = open(embedding_path,'r',encoding='utf8')
    for line in tqdm(f):
        value = line.split(' ')
        word = value[0]
        coef = np.array(value[1:], dtype='float32')
        embedding_vector[word] = coef
    f.close()
    return embedding_vector

if __name__ == '__main__':

    model_names = ['VGG16','VGG19','ResNet50','InceptionV3','Xception']
    dataset_name = 'dataset'  #'silver_negative' 'silver_positive'
    if dataset_name == 'dataset':
        im_path = 'data/dataset_images/dataset_image/'
    elif dataset_name == 'silver_negative':
        im_path = 'data/silver_negative/'
    elif dataset_name == 'silver_positive':
        im_path = 'data/silver_positive/'

    embedding_vector = embedding_load('embedding_300d.txt')

    lbp_feature_dict = {}
    other_feature_dict = {}
    tags_embedding_feature_dict = {}
    #last_layer_feature_dict = {}
    #color_hist_feature_dict = {}

    for model_name in model_names:
        out_tag_file = open(dataset_name+'_'+ model_name + '_image_tags.txt', 'w', encoding='utf8')
        deep_learning_feature_file_name = 'feature_data/' + dataset_name + '_'+ model_name +'_image_tag_feature.npy'

        if model_name == 'VGG16':
            model = VGG16(weights='imagenet', include_top=True)
            im_size = 224
        elif model_name == 'VGG19':
            model = VGG19(weights='imagenet', include_top=True)
            im_size = 224
        elif model_name == 'ResNet50':
            model = ResNet50(weights='imagenet', include_top=True)
            im_size = 224
        elif model_name == 'InceptionV3':
            model = InceptionV3(weights='imagenet', include_top=True)
            im_size = 299
        elif model_name == 'Xception':
            model = Xception(weights='imagenet', include_top=True)
            im_size = 299
        #print(model.summary())

        for im in os.listdir(im_path):
            print(im)
            try:
                img = Image.load_img(im_path + im, target_size=(im_size, im_size))
            except:
                print(im_path + im)
                continue
            x = Image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            # im_last_layer_feature = image_feature_extraction(x, model)
            # print('im_last_layer_feature length ', len(im_last_layer_feature))
            image_tags = image_tags_extraction(x, model)
            tags = ''
            for tag in image_tags:
                # print(tag[1])
                tags = tags + tag[1] + ' '
            print(im + '\t' + tags + '\n')
            out_tag_file.write(im + '\t' + tags + '\n')

            tags_embedding_feature = im_tags_embedding(image_tags, embedding_vector)

            tags_embedding_feature_dict[im] = tags_embedding_feature

        np.save(deep_learning_feature_file_name, tags_embedding_feature_dict)
        out_tag_file.close()


    for im in os.listdir(im_path):
            print(im)
            im_size = os.path.getsize(im_path+im)
            print('im_size:', im_size)
            image = cv2.imread(im_path+im)
            try:
                dominant_color = dominant_color_rgb(image)
                print('dominant_color:', dominant_color[0])
            except:
                dominant_color = np.array([[0,0,0]])

            colorfulness = image_colorfulness(image)
            print('colorfulness:', colorfulness)

            sp = image.shape
            high = sp[0]
            width = sp[1]
            print('sp',sp[2])
            total = 0
            b = 45
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #print(gray)
            arr = np.array(image)
            for h in range(arr.shape[0]):
                for w in range(arr.shape[1]):
                    total += (arr[(h,w,0)]-b)*(arr[(h,w,0)]-b)

            contast = total/high*width
            print(contast)
            if contast>0:
                contrast = math.sqrt(contast)
            else:
                contast = contast*(-1)
                contrast = math.sqrt(contast)*(-1)

            print('contrast:', contrast)

            desc = LocalBinaryPatterns(8, 1.0)  # 59
            hist_LBP = desc.describe(gray)  #
            print(hist_LBP)

            #color_hist = im_color_hist(image)  # 768
            #color_hist.append(h)
            #color_hist.append(w)

            lbp_feature_dict[im] = list(hist_LBP)
            other_feature_dict[im] = [im_size/1000, high, width, colorfulness, contrast/1000]+list(dominant_color[0])
            #print([im_size/1000, high, width, colorfulness, contrast/1000]+list(dominant_color[0]))

            #color_hist_feature_dict[im] = color_hist
            #last_layer_feature_dict[im] = im_last_layer_feature

    np.save('feature_data/' + dataset_name+'_image_LBP_feature.npy', lbp_feature_dict)
    np.save('feature_data/' + dataset_name+'_image_other_feature.npy', other_feature_dict)
    #np.save(dataset_name+'_image_color_feature.npy', color_hist_feature_dict)

