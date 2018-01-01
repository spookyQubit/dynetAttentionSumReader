import os
import nltk
import logging
from collections import Counter
from collections import defaultdict
import pickle
import dill

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


def get_testing_logger():
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d-%m-%Y:%H:%M:%S',
                        filename='testing/example.log',
                        level=logging.INFO)
    return logging.getLogger('data_utils')


class CBTData(object):
    def __init__(self,
                 vocab_file=None,
                 w2i_file=None,
                 bos_tag='<s>',
                 eos_tag='</s>',
                 unk_tag='</unk>',
                 max_data_points=None,
                 log_every_n_data_points=5000,
                 query_in_line_number=21,
                 logger=None):

        self.vocab_file = vocab_file
        self.w2i_file = w2i_file
        self.bos_tag = bos_tag
        self.eos_tag = eos_tag
        self.unk_tag = unk_tag
        self.vocabulary = []
        self.w2i = {}  # will be defaultdict
        self.max_data_points = max_data_points
        self.log_every_n_data_points = log_every_n_data_points
        self.query_in_line_number = query_in_line_number
        self.logger = logger

    def build_new_vocabulary_and_save(self,
                                      data_files,
                                      vocab_file,
                                      keep_top_vocab_percent):

        # get the vocabulary from data_file
        self.vocabulary = self._get_vocabulary_from_files(data_files,
                                                          keep_top_vocab_percent,
                                                          self.max_data_points)

        # Save the vocabulary in vocab_file
        self.logger.info("Saving vocab to file {}".format(self.vocab_file))
        self.vocab_file = vocab_file
        # clear the content of a previous vocab file if it exists
        if os.path.exists(self.vocab_file):
            open(self.vocab_file, 'w').close()

        with open(self.vocab_file, 'w+') as f:
            pickle.dump(self.vocabulary, f)
        self.logger.info("Done saving vocab to file")

    def build_new_w2i_from_existing_vocab_and_save(self, w2i_file):

        """
        # Build the word --> index using the self.vocabulary
        """
        self.w2i = defaultdict(lambda: self.vocabulary.index(self.unk_tag))
        for i, w in enumerate(self.vocabulary):
            self.w2i[w] = i

        # save the w2i file
        self.logger.info("Saving w2i to file {}".format(self.w2i_file))
        self.w2i_file = w2i_file
        # clear the content of a previous vocab file if it exists
        if os.path.exists(self.w2i_file):
            open(self.w2i_file, 'w').close()

        with open(self.w2i_file, "w+") as f:
            pickle.dump(self.w2i, f)
        self.logger.info("Done saving w2i to file")

    def get_vocab(self):
        if self.vocabulary is None:
            self.logger.error("Vocabulary is not yet initialized")
            raise ValueError
        else:
            return self.vocabulary

    def get_w2i(self):
        if self.w2i is None:
            self.logger.error("w2i is not yet initialized")
            raise ValueError
        else:
            return self.w2i

    def get_data_and_save(self, data_files, data_save_file):
        X, y = self.get_data(data_files)
        with open(data_save_file, "w+") as f:
            pickle.dump({"X": X, "y": y}, f)
        self.logger.info("Done saving data to file")
        return X, y

    def get_data(self, data_files):

        max_data_points = self.max_data_points

        X = []
        answers = []
        data_points = 0

        for file_path in data_files:
            if not os.path.exists(file_path):
                raise ValueError

            if self.max_data_points and \
                            data_points >= self.max_data_points:
                break

            current_context = []
            with open(file_path, "rb") as f:
                for line_number, line in enumerate(f):

                    if self.max_data_points and \
                                    data_points >= self.max_data_points:
                        break

                    if (line_number + 1) % (self.query_in_line_number + 1) == 0:
                        # Every data points are separated by a blank line
                        continue
                    elif (line_number + 1) % (self.query_in_line_number + 1) == self.query_in_line_number:
                        tokenized_question, answer, candidate_answers = self._process_query_line(line)

                        x = {"context": current_context,
                             "question": tokenized_question,
                             "candidates": candidate_answers}

                        if not self._is_ok(x, answer):
                            continue

                        X.append(x)
                        answers.append(answer)
                        current_context = []

                        data_points += 1
                        if data_points % self.log_every_n_data_points == 0:
                            self.logger.info("get_data: completed = {} data_points".format(data_points))
                    else:
                        tokenized_sentence = self._process_sentence(line)
                        current_context.extend(tokenized_sentence)

        return X, answers

    def load_vocabulary(self, vocab_file):
        if not os.path.exists(vocab_file):
            self.logger.error("{} vocab file does not exist".format(vocab_file))
            raise ValueError

        with open(vocab_file, "rb") as f:
            self.vocabulary = pickle.load(f)
            self.logger.info("Vocabulary size = {}".format(len(self.vocabulary)))
            if len(self.vocabulary) == 0:
                self.logger.error("{} vocab file is empty".format(vocab_file))
                raise ValueError
        return

    def load_w2i(self, w2i_file):
        if not os.path.exists(w2i_file):
            self.logger.error("{} w2i file does not exist".format(w2i_file))
            raise ValueError

        with open(w2i_file, "rb") as f:
            self.w2i = pickle.load(f)
            self.logger.info("w2i size = {}".format(len(self.w2i)))
            if len(self.w2i) == 0:
                self.logger.error("{} w2i file is empty".format(w2i_file))
                raise ValueError
        return

    def load_data(self, data_save_file):
        if not os.path.exists(data_save_file):
            self.logger.error("{} data_save_file does not exist".format(data_save_file))
            raise ValueError

        with open(data_save_file, "rb") as f:
            d = pickle.load(f)
            if "X" not in d:
                self.logger.error("{} the data file does not contain X!".format(data_save_file))
                raise ValueError
            if "y" not in d:
                self.logger.error("{} the data file does not contain y!".format(data_save_file))
                raise ValueError

            X = d["X"]
            y = d["y"]

            if len(X) != len(y):
                self.logger.error("X and y have different data sizes in file {}!".format(data_save_file))
                raise ValueError

            self.logger.info("Number of data points loaded = {}".format(len(y)))
        return X, y

    def _is_ok(self, x, y):
        context = x["context"]
        question = x["question"]
        candidates = x["candidates"]
        answer = y

        # check that the candidates are a part of the context
        for candidate in candidates:
            candidate_indices = [i for i, x in enumerate(context) if x == candidate]
            if len(candidate_indices) == 0:
                self.logger.info("is_ok = False as candidate not in context")
                return False

        # check that the answer is a part of the context
        is_answer_part_of_candidates = False
        for candidate in candidates:
            if candidate is answer:
                is_answer_part_of_candidates = True
                break

        # "and" all the previous checkes
        # no need to "and" for candidate checking as False is directly returned
        is_ok = is_answer_part_of_candidates
        return is_ok

    def _get_vocabulary_from_files(self, data_files, keep_top_vocab_percent, max_data_points):
        vocab = Counter()
        data_points = 0
        for file_path in data_files:
            if not os.path.exists(file_path):
                self.logger.error("file_path = {} to create vocab does not exist".format(file_path))
                raise ValueError

            if (max_data_points is not None) and \
                    (data_points >= max_data_points):
                break

            with open(file_path, "rb") as f:
                for line_number, line in enumerate(f):

                    if (max_data_points is not None) and \
                            (data_points >= max_data_points):
                        break
                    if (line_number + 1) % (self.query_in_line_number + 1) == 0:
                        # Every data points are separated by a blank line
                        continue
                    elif (line_number + 1) % (self.query_in_line_number + 1) == self.query_in_line_number:
                        # the dictionary should not be updated with candidate_answers
                        # as they already for the part of the context.
                        tokenized_question, answer, _ = self._process_query_line(line)
                        vocab.update(tokenized_question)
                        vocab.update([answer])
                        data_points += 1
                        if data_points % self.log_every_n_data_points == 0:
                            self.logger.info("_build_vocab: completed = {} data_points".format(data_points))
                    else:
                        tokenized_sentence = self._process_sentence(line)
                        vocab.update(tokenized_sentence)


        """
        if max_vocab_size is not None:
            frequent_words = vocab.most_common(max_vocab_size)
        else:
            frequent_words = vocab.most_common()

        frequent_words = dict(frequent_words).keys()
        """

        frequent_words = vocab.most_common(int(len(vocab) * keep_top_vocab_percent / float(100)))
        frequent_words = dict(frequent_words).keys()

        if self.unk_tag is not None:
            frequent_words.append(self.unk_tag)

        return frequent_words

    def _process_query_line(self, query_line):
        """
        #:param query_line: Question followed by answer which is
        #                   followed by the candidate answers
        #                   I think I 've told you that his name was XXXXX -- did I not ?
        #                   Prigio
        #                   CHAPTER|Flitter|Prigio|Saracens|
        #                   lumber-room|ogres|p9.jpg|rustling|
        #                   speaking-trumpet|wishing-cap
        #:return: tokenized question, answer, tokenized candidates
        """
        [question, answer, _, candidates] = query_line.split("\t")

        # process question
        question = self._process_sentence(question)

        # process candidates
        candidates = candidates.strip("\n").split("|")
        # Make the true answer to be the first of the candidates
        candidates.remove(answer)
        candidates.insert(0, answer)
        for n, cand in enumerate(candidates):
            if cand is "":
                candidates[n] = self.unk_tag
        return question, answer, candidates

    def _process_sentence(self, sentence):
        """
        #:param line: A sentence
        #:return: Returns a list of words and removes the
        #         beginning number which is present at the
        #         beginning of each sentence.
        #:example: sentence = "Winter is coming .
        #:return: ["<S>", "Winter", "is", "coming", "</S>"]
        """

        sentence = nltk.word_tokenize(sentence)[1:]
        if self.bos_tag:
            sentence.insert(0, self.bos_tag)
        if self.eos_tag:
            sentence.append(self.eos_tag)
        return sentence


def testing():
    testing_dir = "testing"
    train_file = "testing_train.txt"
    valid_file = "testing_valid.txt"

    train_files = [train_file]
    valid_files = [valid_file]

    train_files = [os.path.join(testing_dir, f) for f in train_files]
    valid_files = [os.path.join(testing_dir, f) for f in valid_files]

    logger = get_testing_logger()
    logger.debug(" train_files = {} "
                 "\n valid_files = {} ".format(train_files, valid_files))

    vocab_file = os.path.join(testing_dir, "vocab.txt")
    w2i_file = os.path.join(testing_dir, "w2i.txt")

    cbt_data = CBTData(vocab_file=vocab_file,
                       w2i_file=w2i_file,
                       query_in_line_number=21,
                       max_data_points=2,
                       logger=logger)
    cbt_data.build_new_vocabulary_and_save(train_files, vocab_file, keep_top_vocab_percent=90)
    cbt_data.build_new_w2i_from_existing_vocab_and_save(w2i_file)

    logger.debug("vocab = {}".format(cbt_data.get_vocab()))
    logger.debug("w2i = {}".format(cbt_data.get_w2i()))
    logger.info("len(vocab) = {}".format(len(cbt_data.get_vocab())))

    logger.debug("Training data = {}".format(cbt_data.get_data(train_files)))
    logger.debug("Validation data = {}".format(cbt_data.get_data(valid_files)))


def main():
    c = Counter()
    c.update(["s", "h", "s", "d", "d", "d", "f", "l"])
    testing()

if __name__ == "__main__":
    main()