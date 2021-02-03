import torch
import bert
from tensorflow import keras
from transformers import BertweetModel, BertweetTokenizer, AutoTokenizer, AutoModel, RobertaConfig
from fairseq.models.roberta import RobertaModel
import numpy as np
from nltk.tokenize import TweetTokenizer
from emoji import demojize
import re,os
import bert
import pandas as pd
import sentencepiece as spm
import aidrtokenize as aidrtokenize
import emoji

tokenizer = TweetTokenizer()


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

def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token

def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = normTweet.replace("cannot ", "can not ").replace("n't ", " n't ").replace("n 't ", " n't ").replace(
        "ca n't", "can't").replace("ai n't", "ain't")
    normTweet = normTweet.replace("'m ", " 'm ").replace("'re ", " 're ").replace("'s ", " 's ").replace("'ll ",
                                                                                                         " 'll ").replace(
        "'d ", " 'd ").replace("'ve ", " 've ")
    normTweet = normTweet.replace(" p . m .", "  p.m.").replace(" p . m ", " p.m ").replace(" a . m .",
                                                                                            " a.m.").replace(" a . m ",
                                                                                                             " a.m ")

    normTweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normTweet)

    return " ".join(normTweet.split())

def Bertolo_feature_extraction(ids,texts, feature_file_name):
    config = RobertaConfig.from_pretrained("./bert-like models/bertolo/config.json")
    tokenizer1 = AutoTokenizer.from_pretrained("./bertolo",normalization=True)
    model = AutoModel.from_pretrained("./bertolo",config=config)

    feature_dict={}
    for i in range(len(ids)):
        id = ids[i]
        print(id)
        title = texts[i]
        #input_ids = torch.tensor([tokenizer.encode(tumblr_text)])
        input_ids = tokenizer1.encode(title, return_tensors="pt")
        print(input_ids)

        #with torch.no_grad():
        features = model(input_ids)[0]  # Models outputs are now tuples
        print(features.size())
        feature = torch.mean(features, 1, True).detach().numpy()
        print(feature[0])

        feature = list(feature[0][0])
        print(feature)
        print(len(feature))
        feature_dict[tumblr_id]=feature
    np.save(feature_file_name, feature_dict)


def Roberta_feature_extraction(ids,texts,feature_file_name):
    roberta = RobertaModel.from_pretrained('roberta.large',checkpoint_file = 'model.pt')
    roberta.eval()

    feature_dict={}
    for i in range(len(ids)):
        id = ids[i]
        print(id)
        title = texts[i]
        tokens = roberta.encode(title)
        #assert tokens.tolist() == [0, 31414, 232, 328, 2]
        print(tokens.tolist())
        roberta.decode(tokens)  # 'Hello world!'

        # Extract the last layer's features
        last_layer_features = roberta.extract_features(tokens)
        #assert last_layer_features.size() == torch.Size([1, 5, 1024])
        print(torch.mean(last_layer_features,1,True))
        #print(last_layer_features.detach().numpy().shape())
        print(len(torch.mean(last_layer_features,1,True).detach().numpy().tolist()[0][0]))
        #print(np.mean(last_layer_features.detach().numpy(), axis=0).tolist()[0])
        print(torch.mean(last_layer_features, 1, True).detach().numpy().tolist()[0][0])

        feature_dict[tumblr_id]=torch.mean(last_layer_features, 1, True).detach().numpy().tolist()[0][0]
    np.save(feature_file_name, feature_dict)

def Albert_model(max_seq_len):
    model_name = "albert_large"
    model_dir = bert.fetch_tfhub_albert_model(model_name, ".models")
    model_params = bert.albert_params(model_name)
    model_params.shared_layer = True
    model_params.embedding_size = 1024

    l_bert = bert.BertModelLayer.from_params(model_params, name="albert")

    l_input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32')

    # using the default token_type/segment id 0
    output = l_bert(l_input_ids)                              # output: [batch_size, max_seq_len, hidden_size]
    output = keras.layers.GlobalAveragePooling1D()(output)
    model = keras.Model(inputs=l_input_ids, outputs=output)
    model.build(input_shape=(None, max_seq_len))
    # use in a Keras Model here, and call model.build()
    bert.load_albert_weights(l_bert, model_dir)       # should be called after model.build()
    return model, model_dir

def Albert_feature_extraction(ids,texts,max_seq_len,feature_file_name):
    model, model_dir = Albert_model(max_seq_len)
    spm_model = os.path.join(model_dir, "assets", "30k-clean.model")
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model)
    do_lower_case = True

    feature_dict={}

    for i in range(len(ids)):
        id = ids[i]
        print(id)
        title = texts[i]
        processed_text = bert.albert_tokenization.preprocess_text(title, lower=do_lower_case)
        processed_text = "[CLS] "+processed_text+" [SEP]"
        print(processed_text)
        token_ids = bert.albert_tokenization.encode_ids(sp, processed_text)
        while len(token_ids) < max_seq_len:
            token_ids.append(0)
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
        print(token_ids)
        print(type(token_ids))
        feature = model.predict(np.array([token_ids]))
        feature_dict[id] = feature.tolist()[0]
    np.save(feature_file_name, feature_dict)


def Bertweet_feature_extraction(ids,texts,max_seq_len,feature_file_name):
    bertweet = BertweetModel.from_pretrained("vinai/bertweet-base")
    tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

    feature_dict={}
    for i in range(len(ids)):
        id = ids[i]
        print(id)
        title = texts[i]
        print(title)
        input_ids = torch.tensor([tokenizer.encode(title)])
        with torch.no_grad():
            features = bertweet(input_ids)  # Models outputs are now tuples

        feature_dict[id]=list(features[1][0].numpy())
    np.save(feature_file_name, feature_dict)


def Bert_feature_extraction(ids,texts, max_seq_len, feature_file_name):
    #https://github.com/kpe/bert-for-tf2
    model_dir = ".models/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12"
    bert_ckpt_file = os.path.join(model_dir, "bert_model.ckpt")

    bert_params = bert.params_from_pretrained_ckpt(model_dir)
    l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

    l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')
    l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')

    # using the default token_type/segment id 0
    output = l_bert(l_input_ids)  # output: [batch_size, max_seq_len, hidden_size]

    output = keras.layers.GlobalAveragePooling1D()(output)
    model = keras.Model(inputs=l_input_ids, outputs=output)
    model.build(input_shape=(None, max_seq_len))

    bert.load_stock_weights(l_bert, bert_ckpt_file)

    do_lower_case = not (model_dir.find("cased") == 0 or model_dir.find("multi_cased") == 0)
    bert.bert_tokenization.validate_case_matches_checkpoint(do_lower_case, bert_ckpt_file)
    vocab_file = os.path.join(model_dir, "vocab.txt")
    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

    feature_dict = {}
    for i in range(len(ids)):
        id = ids[i]
        print(id)
        title = texts[i]
        tokens = tokenizer.tokenize(title)
        print(tokens)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        while len(token_ids) < max_seq_len:
            token_ids.append(0)
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
        print(token_ids)

        token_ids = np.array([token_ids], dtype=np.int32)

        feature = model.predict(token_ids)

        feature_dict[id] = feature.tolist()[0]

    np.save(feature_file_name,feature_dict)

if __name__ == '__main__':
    max_seq_len = 256     #128
    dataset_name = 'dataset'  #silver
    feature_type = 'bertweet'  #'bert', 'albert', 'roberta', 'bertolo'
    feature_file_name = 'feature_data/' + dataset_name + '_'+feature_type+'_'+str(max_seq_len)+'_feature.npy'

    if dataset_name == 'silver':
        data_path = 'data/tu_all_data.tsv'
        data = pd.read_csv(data_path, sep='\t', header=0, encoding="ISO-8859-1")
        ids = data["id"].tolist()
        texts = data["content"].tolist()
    elif dataset_name == 'dataset':
        data_path = 'data/dataset_texts/dataset_all_text.txt'
        ids = []
        texts = []
        with open(data_path, 'r', encoding="utf8") as f:
            for line in f.readlines():
                tumblr = line.replace('\n', '')[1:-1]  # 138051004348
                tumblr_id = tumblr[1:19]
                # print(tumblr_id)
                tumblr_text = str(tumblr[23:-4])
                tumblr_text = text_proprecess(tumblr_text)
                ids.append(tumblr_id)
                texts.append(tumblr_text)
    if len(ids) == len(texts):
        if feature_type == 'bertweet':
            Bertweet_feature_extraction(ids,texts, max_seq_len, feature_file_name)
        elif feature_type == 'albert':
            Albert_feature_extraction(ids,texts,max_seq_len,feature_file_name)
        elif feature_type == 'bert':
            Bert_feature_extraction(ids,texts,max_seq_len,feature_file_name)
        elif feature_type == 'roberta':
            Roberta_feature_extraction(ids, texts, feature_file_name)
        elif feature_type == 'bertolo':
            Bertolo_feature_extraction(ids,texts, feature_file_name)