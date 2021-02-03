import pandas as pd
import nlp_features.nlp_features.nlp_features as nlp_feat
import affective_features.affective_features as affective_features
import numpy as np

def text_feature_extraction(text, tag):
    print(text)
    features = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    aff_features = [1, 1, 1, 1, 1, 1, 1]
    nlp = nlp_feat.start()
    nlp_affective = affective_features.start()
    try:
        if len(tag)<2:
              tag = '0'
    except:
        tag = '0'
    print(tag)
    david_feature = nlp_feat.extract(nlp, features, text, tag)
    affective_feat = affective_features.extract(nlp_affective, aff_features, text)

    return affective_feat, david_feature

if __name__ == '__main__':
    dataset_name = 'dataset'

    if dataset_name == 'dataset':
        data_path = 'data/dataset_texts/dataset_all_text.txt'
        ids = []
        texts = []
        with open(data_path, 'r', encoding="utf8") as f:
            for line in f.readlines():
                tumblr = line.replace('\n', '')[1:-1]  # 138051004348
                tumblr_id = tumblr[1:19]
                tumblr_text = str(tumblr[23:-4])
                ids.append(tumblr_id)
                texts.append(tumblr_text)
    if dataset_name == 'silver':
        data_path = 'data/tu_all_data.tsv'
        data = pd.read_csv(data_path, sep='\t', header=0, encoding="ISO-8859-1")
        ids = data["id"].tolist()
        texts = data["content"].tolist()

    affective_feature_dict = {}
    nlp_feature_dict = {}
    if len(ids) == len(texts):
        for i in range(len(ids)):
            tumblr_id = ids[i]
            tumblr_text = texts[i]
            print(tumblr_text)
            affective_feature, david_feature = text_feature_extraction(tumblr_text, '')
            print(len(affective_feature))
            print(len(david_feature))
            print(david_feature)
            affective_feature_dict[tumblr_id] = affective_feature
            nlp_feature_dict[tumblr_id] = david_feature

    np.save('feature_data/' + dataset_name + '_affective_feature.npy', affective_feature_dict)
    np.save('feature_data/' + dataset_name + '_nlp_feature.npy', nlp_feature_dict)
