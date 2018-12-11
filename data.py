import os
import pickle
import random
import codecs
import numpy as np

tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }

# 默认数据集 MSRA tags, BIO
tag2label_msra = {"O": 0,
                  "B-PER": 1, "I-PER": 2,
                  "B-LOC": 3, "I-LOC": 4,
                  "B-ORG": 5, "I-ORG": 6
                  }

# 人民日报数据集
tag2label_chinadaily = {"O": 0,
                        "B-PERSON": 1, "I-PERSON": 2,
                        "B-LOC": 3, "I-LOC": 4,
                        "B-ORG": 5, "I-ORG": 6,
                        "B-GPE": 7, "I-GPE": 8,
                        "B-MISC": 9, "I-MISC": 10
                        }
# Weibo_NER
tag2label_weibo_ner = {"O": 0,
                       "B-PER.NAM": 1, "I-PER.NAM": 2,
                       "B-LOC.NAM": 3, "I-LOC.NAM": 4,
                       "B-ORG.NAM": 5, "I-ORG.NAM": 6,
                       "B-GPE.NAM": 7, "I-GPE.NAM": 8,
                       "B-PER.NOM": 9, "I-PER.NOM": 10,
                       "B-LOC.NOM": 11, "I-LOC.NOM": 12,
                       "B-ORG.NOM": 13, "I-ORG.NOM": 14
                       }

# Resume_NER
tag2label_resume_ner = {"O": 0,
                        "B-NAME": 1, "M-NAME": 2, "E-NAME": 3, "S-NAME": 4,
                        "B-RACE": 5, "M-RACE": 6, "E-RACE": 7, "S-RACE": 8,
                        "B-CONT": 9, "M-CONT": 10, "E-CONT": 11, "S-CONT": 12,
                        "B-LOC": 13, "M-LOC": 14, "E-LOC": 15, "S-LOC": 16,
                        "B-PRO": 17, "M-PRO": 18, "E-PRO": 19, "S-PRO": 20,
                        "B-EDU": 21, "M-EDU": 22, "E-EDU": 23, "S-EDU": 24,
                        "B-TITLE": 25, "M-TITLE": 26, "E-TITLE": 27, "S-TITLE": 28,
                        "B-ORG": 29, "M-ORG": 30, "E-ORG": 32, "S-ORG": 33,
                        }

tag2label_mapping = {
    'MSRA': tag2label_msra,
    '人民日报': tag2label_chinadaily,
    'Weibo_NER': tag2label_weibo_ner,
    'ResumeNER': tag2label_resume_ner

}


def build_character_embeddings(pretrained_emb_path, embeddings_path, word2id, embedding_dim):
    print('loading pretrained embeddings from {}'.format(pretrained_emb_path))
    pre_emb = {}
    for line in codecs.open(pretrained_emb_path, 'r', 'utf-8'):
        line = line.strip().split()
        if len(line) == embedding_dim + 1:
            pre_emb[line[0]] = [float(x) for x in line[1:]]
    word_ids = sorted(word2id.items(), key=lambda x: x[1])
    characters = [c[0] for c in word_ids]
    embeddings = list()
    for i, ch in enumerate(characters):
        if ch in pre_emb:
            embeddings.append(pre_emb[ch])
        else:
            embeddings.append(np.random.uniform(-0.25, 0.25, embedding_dim).tolist())
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save(embeddings_path, embeddings)


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data


def vocab_build(vocab_path, corpus_path, min_count=1):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id) + 1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

# if __name__ == '__main__':
#     word2id = read_dictionary(os.path.join('data_path', 'MSRA', 'word2id.pkl'))
#     build_character_embeddings('./sgns.wiki.char', './vectors.npy', word2id, 300)
