import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
os.sys.path.extend(['D:\\a_github项目\\ASSORT-Automatic-Summarization-of-Stack-Overflow-Posts-main\\model'])
from transformers import AutoModel, AutoTokenizer
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from classification_head import *


def entity_overlap(sentence, tags, question_title):
    tags = literal_eval(tags)
    total = len(tags)
    overlap = 0
    for i in tags:
        if question_title.lower().find(i) != -1 and sentence.lower().find(i) != -1:
            overlap += 1
    if total != 0:
        return float(overlap) / total
    else:
        return 0


def comparative_adjective(tags):
    result = 0
    for x, y in tags:
        if y == "JJR":
            result = 1
    return result


def superlative_adjective(tags):
    result = 0
    for x, y in tags:
        if y == "JJS":
            result = 1
    return result


def imperative_sentence(tags):
    if tags[0][1] == "VB":
        return 1
    else:
        return 0


def pattern_matching(sentence):
    sentence = sentence.lower()
    result = [0] * 19
    patterns = ["however,", 'first,', 'in short,', 'in this case,', 'in general,', 'on the other hand,', 'then,', \
                'alternatively,', 'in other words,', 'in addition,', 'in practice,', 'in fact,', 'otherwise,',
                'if you care,',
                'in contrast,', 'finally,', 'below is', 'additionally,', 'furthermore,']
    for index, i in enumerate(patterns):
        if sentence.find(i) != -1:
            result[index] = 1
    return result


# 根据给定的句子、位置、标签和问题标题，提取一些明确的特征。
def explicit_features(i, index, tags, question_title):
    results = []
    truth = []
    ef_for_one_sentence = [0] * 9
    ef_for_one_sentence[0] = entity_overlap(i, tags, question_title)
    if index < 3:
        ef_for_one_sentence[1] = 1
    if i.find("<code") != 0:
        ef_for_one_sentence[3] = 1
    if i.find("<br") != 0:
        ef_for_one_sentence[4] = 1
    if i.find('<li') != 0:
        ef_for_one_sentence[5] = 1

    # Comparative/Superlative adjectives, imperative sentence
    words = nltk.word_tokenize(i)
    tagged = nltk.pos_tag(words)
    ef_for_one_sentence[6] = comparative_adjective(tagged)
    ef_for_one_sentence[7] = superlative_adjective(tagged)
    ef_for_one_sentence[8] = imperative_sentence(tagged)
    # Phrase matching
    ef_for_one_sentence += pattern_matching(i)
    return ef_for_one_sentence


import pandas as pd
from ast import literal_eval

# 注意，old_seentence.csv没有这个文件

#old_sentences = pd.read_csv("old_sentences.csv")
import numpy as np
from tqdm import trange


# 获得bertOverflow生产的向量
def get_representation(df, tokenizer, model):
    device = torch.device('cuda:0')
    model=model.to(device)
    explicit_features_for_df = np.empty((0, 28), dtype=float, order='C')
    for index, i in enumerate(df.sentence):
        position = df.position[index]
        tags = df.tags[index]
        question_title = df.question_title[index]
        explicit_features_for_df = np.append(explicit_features_for_df,
                                             np.array(explicit_features(i, position, tags, question_title)))
    explicit_features_for_df = explicit_features_for_df.reshape(-1, 28)

    sentence_bert_for_df = np.empty((0, 768), dtype=float, order='C')

    for i in trange(len(df.sentence)):
        i = df.sentence[i]
        encoded_input = tokenizer(i, return_tensors='pt').to(device)
        output = model(**encoded_input)
        sentence_bert_for_df = np.append(sentence_bert_for_df, output['pooler_output'].cpu().detach().numpy(), axis=0)

    return np.concatenate((sentence_bert_for_df, explicit_features_for_df), axis=1)


# BERTOverflow

#boverflow_tokenizer = AutoTokenizer.from_pretrained("jeniya/BERTOverflow")
#boverflow_model = AutoModel.from_pretrained("jeniya/BERTOverflow")
boverflow_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
boverflow_model = AutoModel.from_pretrained("bert-base-uncased")
dataset = pd.read_csv("D://a_github项目//ASSORT-Automatic-Summarization-of-Stack-Overflow-Posts-main//data//dataset.csv")
# bertoverflow的向量
old_boverflow_embeddings = get_representation(dataset, boverflow_tokenizer, boverflow_model)



# Train conceptual sentence classifier
print("training  sentence classifier")
from sklearn.model_selection import train_test_split


# old_sentence 文件缺失，只是每一个句子的真伪值无法判断
# 更改 train_nn,以支持在gpu上运行
X_train, X_test, y_train_conceptual, y_test_conceptual = train_test_split(
    old_boverflow_embeddings,
    list(dataset.truth), test_size=0.2, random_state=42)
#train_nn(X_train, y_train_conceptual, X_test, y_test_conceptual, "overall", lr=1e-5, epoch=500)
train_nn(X_train, y_train_conceptual, X_test, y_test_conceptual, "overall", lr=1e-5, epoch=800)


# Constructing test_set
test_set = np.concatenate((y_train_conceptual, y_train_howto, y_train_debug), axis=0)
test_truth = np.concatenate((y_test_conceptual, y_test_howto, y_test_debug), axis=0)

# 加载预训练的question 分类器，这部分目前没见到？
print("Loading pre-trained question classifier...")
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
clf = pickle.load(open('question_classifier.sav', 'rb'))

print("Loading sentence-classifiers we just trained...")
# Howto model
size = 796
howtomodel = Net(size)
howtomodel.load_state_dict(torch.load('nn_old_approach_50_how_to_best.pt'))
# Conceptual model
conceptualmodel = Net(size)
conceptualmodel.load_state_dict(torch.load('nn_old_approach_50_conceptual_best.pt'))
# Debug model
debugmodel = Net(size)
debugmodel.load_state_dict(torch.load('nn_old_approach_50_debug_best.pt'))

print("Getting bag-of-word representations of question titles")
new_bag_of_words = vectorizer.transform(dataset.question_title).toarray()
probs = clf.predict_proba(new_bag_of_words)

print("Applying different sentence classifiers...")


def get_sm_from_sentence_classifier(model, embeddings):
    m = nn.Softmax(dim=1)
    scores = model(torch.tensor(embeddings).float())
    return m(scores).detach().numpy()[:, 1]


new_howtosm = get_sm_from_sentence_classifier(howtomodel, test_set)
new_conceptualsm = get_sm_from_sentence_classifier(conceptualmodel, test_set)
new_debugsm = get_sm_from_sentence_classifier(debugmodel, test_set)

new_scores_from_sentence_classifier = np.concatenate(
    (new_howtosm.reshape(1, -1), new_conceptualsm.reshape(1, -1), new_debugsm.reshape(1, -1)), axis=0)

new_final_score = probs.T * new_scores_from_sentence_classifier

new_final_score = new_final_score[0] + new_final_score[1] + new_final_score[2]

print("With threshold 0.5, calculating model performance...")
new_truth = [1 if i > 0.1 else 0 for i in new_final_score]

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

precision = precision_score(dataset.truth.tolist(), new_truth)
recall = recall_score(dataset.truth.tolist(), new_truth)
f1 = 2 * precision * recall / (precision + recall)
print("precision:", precision)
print("recall:", recall)
print("F1:", f1)
