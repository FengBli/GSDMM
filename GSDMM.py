# -*- coding: utf-8 -*-
""" self-implemented Gibbs Sampling for DMM(Dirichlet Multinomial Mixture)

reference:
=========
    Yin, J. and Wang, J., 2014, August. A dirichlet multinomial mixture model-based approach for short text clustering.
    In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 233-242).
    ACM.

"""

import os
import random
import collections

DEBUG = True

class GSDMM(object):
    """ Dirichlet Multinomial Mixture model for short texts """

    n_documents = 0 # $D$
    n_words = 0     # $V$

    doc_ids = []

    word2id_vocabulary = dict()
    id2word_vocabulary = dict()
    documents = []              # list of lists, lists: words' id

    doc_word_occurrence = []    # $N_d^w$, list of dicts, dicts: `word: occurrences` in doc
    doc_word_count = []         # $N_d$

    topic_assignments = []
    topic_doc_count = []        # $m_z$
    topic_word_count = []       # $n_z$

    topic_word_occurrence = []  # $n_z^w$, list of lists

    multi_pros = []

    def __init__(self,
                 corpus_loc,
                 vocab_loc,
                 alpha=0.1,
                 beta=0.01,
                 n_topics=100,
                 n_iterations=20,
                 n_words_each_topic=20):

        self.corpus_loc = corpus_loc
        self.alpha = alpha
        self.beta = beta
        self.n_topics = n_topics
        self.n_iterations = n_iterations
        self.n_words_each_topic = n_words_each_topic

        self.load_vocabulary(vocab_loc)


    def _init(self):
        # initialize m_z, n_z and n_z^w as zero for each cluster z
        self.topic_doc_count = [0 for i in range(self.n_topics)]
        self.topic_word_count = [0 for i in range(self.n_topics)]
        self.topic_word_occurrence = \
            [[0 for i in range(self.n_words)] for i in range(self.n_topics)]

        # randomly sample a cluster for each document
        for doc in self.documents:
            topic = random.randint(0, self.n_topics-1)
            self.topic_assignments.append(topic)
            self.topic_doc_count[topic] += 1

            self.topic_word_count[topic] += len(doc)

            for word in doc:
                self.topic_word_occurrence[topic][word] += 1

        self.multi_pros = [0 for i in range(self.n_topics)]


    def load_vocabulary(self, vocab_loc):
        with open(vocab_loc, 'r') as f:
            for line in f.readlines():
                word, _id = eval(line.strip())

                self.word2id_vocabulary[word] = _id
                self.id2word_vocabulary[_id] = word

        self.n_words = len(self.word2id_vocabulary)
        # if DEBUG:
        #     print("DEBUG: length of vocabulary = {}".format(self.n_words))


    def analyse_corpus(self):
        """ compute :doc_word_count:, :doc_ids:, :documents:,
         :n_documents:, :n_words: by analysing the corpus
        """
        with open(self.corpus_loc, 'r') as fcorpus:
            line = fcorpus.readline()
            while line:
                doc = eval(line.strip())   # dict
                self.doc_ids.append(doc["docid"])

                document = doc["tokenids"]
                self.documents.append(document)

                self.n_words += len(document)
                self.doc_word_count.append(len(document))

                words_occurrence = dict(collections.Counter(document))
                self.doc_word_occurrence.append(words_occurrence)

                line = fcorpus.readline()
        self.n_documents = len(self.documents)
        # if DEBUG:
        #     print("DEBUG: length of corpus = {}".format(self.n_documents))


    def sample_new_topic(self, multi_pros):
        r = random.uniform(0.0, 1.0) * sum(multi_pros)
        _sum = 0.0
        for i in range(len(multi_pros)):
            _sum += multi_pros[i]
            if _sum > r:
                return i
        return len(multi_pros) - 1


    def single_iteration(self, iter):
        if DEBUG:
            print("iteration #{}".format(iter))

        for doc_index in range(self.n_documents):
            doc = self.documents[doc_index]
            topic = self.topic_assignments[doc_index]

            self.topic_doc_count[topic] -= 1
            self.topic_word_count[topic] -= len(doc)


            for word in self.doc_word_occurrence[doc_index]:
                # occurrence = self.doc_word_occurrence[doc_index][word]
                self.topic_word_occurrence[topic][word] -= self.doc_word_occurrence[doc_index][word]

            for t in range(self.n_topics):
                rule1 = (self.topic_doc_count[t] + self.alpha) / \
                       (self.n_documents - 1 + self.n_topics * self.alpha)

                numerator = 1
                denominator = 1
                for i in range(len(doc)):   # start from 0, not 1
                    word = doc[i]
                    denominator *= (self.topic_word_count[t] + self.n_words * self.beta + i)
                    # # each word appears at most once in each document
                    # numerator *= self.topic_word_occurrence[t][word] + self.beta

                    # each word can appear more than once in each document
                    for j in range(self.doc_word_occurrence[doc_index][word]):  # satrt from 0, not 1
                        numerator *= (self.topic_word_occurrence[t][word] + self.beta + j)

                rule2 = (numerator / denominator)

                self.multi_pros[t] = rule1 * rule2

            # sample a new cluster for current :doc:
            topic = self.sample_new_topic(self.multi_pros)

            self.topic_assignments[doc_index] = topic
            self.topic_doc_count[topic] += 1
            self.topic_word_count[topic] += len(doc)
            for word in doc:
                self.topic_word_occurrence[topic][word] += 1


    def inference(self):
        self.analyse_corpus()
        self._init()

        [self.single_iteration(i+1) for i in range(self.n_iterations)]


    def write_topic_assignments(self, fname):
        """ wirte the topic assignment of each document to local file :fname: """

        f = open(fname, 'w')

        [f.write('{{"docid": {}, "cluster": {}}}\n'.format(doc_id, topic)) \
                for doc_id, topic in zip(self.doc_ids, self.topic_assignments)]

        f.close()

    # def write_topic_top_words(self, fname, n_words):
    #     """ write the top :n_words: words of each topic to local file :fname: """
    #     with open(fname, 'w') as f:
    #         for topic in range(n_topics):
    #             word_occurrence = self.topic_word_occurrence[topic]


if __name__ == '__main__':
    model = GSDMM("train_tokens.json", "vocab.json")

    model.inference()
    model.write_topic_assignments("./topic-assignments.json")




