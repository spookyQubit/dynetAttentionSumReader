import dynet as dy
from dynet import GRUBuilder
import logging
import math
import numpy as np


class ASReader(object):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 gru_layers,
                 gru_input_dim,
                 gru_hidden_dim,
                 w2i,
                 adam_alpha=0.01,
                 minibatch_size=16,
                 n_epochs=10,
                 gradient_clipping=10,
                 lookup_init_scale=1.0,
                 logger=None):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.gru_layers = gru_layers
        self.gru_input_dim = gru_input_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.w2i = w2i
        self.adam_alpha = adam_alpha
        self.minibatch_size = minibatch_size
        self.n_epochs = n_epochs
        self.gradient_clipping = gradient_clipping
        self.lookup_init_scale = lookup_init_scale
        self.logger = logger

        assert(self.gru_input_dim == self.embedding_dim)

        self.model, self.context_f_rnn, self.context_b_rnn, self.quest_f_rnn, self.quest_b_rnn, self.model_parameters = self._create_model()

    def fit(self, X, y):
        self.train(X, y, self.w2i, self.gradient_clipping, self.n_epochs, self.minibatch_size)

    def _word_rep(self, w, w2i):
        w_index = w2i[w]
        return self.model_parameters["lookup"][w_index]

    def _get_loss_exp_for_one_instance(self, x, y, w2i):

        context = x["context"]
        question = x["question"]
        candidates = x["candidates"]
        answer = y

        # encode the context
        c_f_init = self.context_f_rnn.initX_trainial_state()
        c_b_init = self.context_b_rnn.initial_state()
        c_wemb = [self._word_rep(w, w2i) for w in context]
        c_f_exps = c_f_init.transduce(c_wemb)
        c_b_exps = c_b_init.transduce(reversed(c_wemb))
        # biGru state for context
        c_bi = [dy.concatenate([f, b]) for f, b in zip(c_f_exps,
                                                       reversed(c_b_exps))]

        # encode the question
        q_f_init = self.quest_f_rnn.initial_state()
        q_b_init = self.quest_b_rnn.initial_state()
        q_wemb = [self._word_rep(w, w2i) for w in question]
        q_f_exps_last = q_f_init.transduce(q_wemb)[-1]
        q_b_exps_last = q_b_init.transduce(reversed(q_wemb))[-1]
        # biGru state for question
        q_bi = dy.concatenate([q_f_exps_last, q_b_exps_last])

        # for each context, calculate the score
        candidate_scores = []
        for candidate in candidates:
            # get all indices of the candidate in the context
            candidate_indices = [i for i, x in enumerate(context) if x == candidate]
            if len(candidate_indices) < 1:
                self.logger.info("context = {} \n candidates = {}".format(context, candidate))
            # calculate the sum of attentions from all the positions where the current candidate occurs
            candidate_score = dy.esum([dy.dot_product(c_bi[i], q_bi) for i in candidate_indices])
            candidate_scores.append(candidate_score)

        candidate_scores_exp = dy.concatenate([score_exp for score_exp in candidate_scores])
        return candidate_scores_exp

    def train(self, X, y, w2i, gradient_clipping_threshold, n_epochs, minibatch_size):
        self.logger.info("Starting to train")

        trainer = dy.AdamTrainer(self.model, self.adam_alpha)
        trainer.set_clip_threshold(gradient_clipping_threshold)
        n_minibatches = int(math.ceil(len(y) / minibatch_size))

        examples_seen = 0
        for epoch in range(n_epochs):
            total_loss = 0.0
            epoch_indices = np.random.permutation(len(y))
            for minibatch in range(n_minibatches):
                batch_indices = epoch_indices[minibatch * minibatch_size:(minibatch + 1) * minibatch_size]

                # Renew the computational graph
                dy.renew_cg()

                y_true_id = 0 # the 0th index in candidates is the true answer
                losses = [dy.pickneglogsoftmax(self._get_loss_exp_for_one_instance(X[batch_indices[i]],
                                                                                   y[batch_indices[i]],
                                                                                   w2i), y_true_id)
                          for i in range(minibatch_size)]
                loss = dy.esum(losses)
                total_loss += loss.value() # forward computation
                loss.backward()
                trainer.update()

                examples_seen += len(batch_indices)
                if minibatch % 10 == 0:
                    self.logger.info('Epoch {}/{}, minibatch = {} , total_loss/examples = {}'.format(epoch + 1,
                                                                                                     self.n_epochs,
                                                                                                     minibatch,
                                                                                                     total_loss/examples_seen))
            #trainer.update_epoch() # this is depricated
            #total_loss /= len(y) # should this be done?
            self.logger.info('Epoch {}/{}, total_loss/examples_seen = {}'.format(epoch + 1, self.n_epochs, total_loss/examples_seen))

        self.logger.info("Done training")

    def predict(self):
        pass

    def load_model(self):
        pass

    def save_model(self):
        pass

    def _create_model(self):
        self.logger.info('Creating the model...')

        model = dy.Model()

        # context gru encoders
        c_fwdRnn = dy.GRUBuilder(self.gru_layers,
                                 self.gru_input_dim,
                                 self.gru_hidden_dim,
                                 model)
        c_bwdRnn = dy.GRUBuilder(self.gru_layers,
                                 self.gru_input_dim,
                                 self.gru_hidden_dim,
                                 model)

        # question gru encoders
        q_fwdRnn = dy.GRUBuilder(self.gru_layers,
                                 self.gru_input_dim,
                                 self.gru_hidden_dim,
                                 model)
        q_bwdRnn = dy.GRUBuilder(self.gru_layers,
                                 self.gru_input_dim,
                                 self.gru_hidden_dim,
                                 model)

        # embedding parameter
        model_parameters = {}
        model_parameters["lookup"] = model.add_lookup_parameters((self.vocab_size,
                                                                  self.gru_input_dim),
                                                                 dy.UniformInitializer(self.lookup_init_scale))
        self.logger.info('Done creating the model')
        return model, c_fwdRnn, c_bwdRnn, q_fwdRnn, q_bwdRnn, model_parameters


class Trainer(object):
    def __init__(self):
        pass
    def train(self):
        pass

