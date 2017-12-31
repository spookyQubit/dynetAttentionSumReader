import dynet as dy
import math
import numpy as np
import operator


class ASReaderTrainer(object):
    def __init__(self,
                 logger):

        self.logger = logger

    def _word_rep(self, w, w2i, lookup_param):
        w_index = w2i[w]
        return lookup_param[w_index]

    def _get_prob_of_each_word_at_every_pos(self, x, w2i, model_params):
        context = x["context"]
        question = x["question"]

        # encode the context
        c_f_init = model_params["c_fwdRnn"].initial_state()
        c_b_init = model_params["c_bwdRnn"].initial_state()
        c_wemb = [self._word_rep(w, w2i, model_params["lookup_params"]) for w in context]
        c_f_exps = c_f_init.transduce(c_wemb)
        c_b_exps = c_b_init.transduce(reversed(c_wemb))
        # biGru state for context
        c_bi = [dy.concatenate([f, b]) for f, b in zip(c_f_exps,
                                                       reversed(c_b_exps))]
        # encode the question
        q_f_init = model_params["q_fwdRnn"].initial_state()
        q_b_init = model_params["q_bwdRnn"].initial_state()
        q_wemb = [self._word_rep(w, w2i, model_params["lookup_params"]) for w in question]
        q_f_exps_last = q_f_init.transduce(q_wemb)[-1]
        q_b_exps_last = q_b_init.transduce(reversed(q_wemb))[0]
        # biGru state for question
        q_bi = dy.concatenate([q_f_exps_last, q_b_exps_last])

        # for each word in the context, calculate its probability to be the answer
        score_of_each_word_at_every_pos = [dy.dot_product(c_bi[i], q_bi) for i in range(len(context))]
        prob_of_each_word_at_every_pos = dy.softmax(dy.concatenate(score_of_each_word_at_every_pos))

        return prob_of_each_word_at_every_pos

    def _get_loss_exp_for_one_instance(self, x, y, w2i, model_params):

        # dynet expression for prob
        prob_of_each_word_at_every_pos = self._get_prob_of_each_word_at_every_pos(x, w2i, model_params)

        # calculate the probability of the correct answer as computed by the current model
        context = x["context"]
        answer = y
        answer_prob_at_all_indices = []
        for word_prob, word in zip(prob_of_each_word_at_every_pos, context):
            if word == answer:
                answer_prob_at_all_indices.append(word_prob)
        loss = -dy.log(dy.esum(answer_prob_at_all_indices))
        return loss

    def _get_minibatch_indices(self, y, epoch_indices, minibatch_size):

        minibatches_in_a_batch = 10
        batch_size = minibatch_size * minibatches_in_a_batch
        n_batches = int(math.ceil(len(y) / batch_size))

        all_minibatch_indices = []
        for batch in range(n_batches):
            batch_indices = epoch_indices[batch * batch_size:(batch + 1) * batch_size]
            sorted_batch_indices = sorted([len(y[i]) for i in batch_indices])

            n_minibatches = int(math.ceil(len(sorted_batch_indices) / minibatch_size))
            for minibatch in range(n_minibatches):
                minibatch_indices = sorted_batch_indices[minibatch * minibatch_size:(minibatch + 1) * minibatch_size]
                all_minibatch_indices.append(minibatch_indices)
        return all_minibatch_indices

    def train(self,
              X, y, w2i,
              model, model_params,
              gradient_clipping_threshold,
              initial_learning_rate,
              n_epochs,
              minibatch_size,
              X_valid=None, y_valid=None,
              n_times_predict_in_epoch=3):

        self.logger.info("Starting to train")

        if len(X) != len(y):
            self.logger.error("X and y do not have same size")

        if X_valid is not None and y_valid is not None:
            if len(X_valid) != len(y_valid):
                self.logger.error("X_valid and y_valid do not have same size")

        trainer = dy.AdamTrainer(model, initial_learning_rate)
        trainer.set_clip_threshold(gradient_clipping_threshold)

        examples_seen = 0
        for epoch in range(n_epochs):

            total_loss = 0.0
            epoch_indices = np.random.permutation(len(y))
            all_minibatch_indices = self._get_minibatch_indices(y, epoch_indices, minibatch_size)

            for minibatch, minibatch_indices in enumerate(all_minibatch_indices):

                # Renew the computational graph
                dy.renew_cg()

                # calculate the loss
                losses = [self._get_loss_exp_for_one_instance(X[minibatch_indices[i]],
                                                              y[minibatch_indices[i]],
                                                              w2i,
                                                              model_params) for i in range(len(minibatch_indices))]
                loss = dy.esum(losses)
                total_loss += loss.value()  # forward computation
                loss.backward()
                trainer.update()

                examples_seen += len(minibatch_indices)
                if minibatch % 10 == 0:
                    self.logger.info('Epoch {}/{}, minibatch = {} , total_loss/examples = {}'.format(epoch + 1,
                                                                                                     n_epochs,
                                                                                                     minibatch,
                                                                                                     total_loss / examples_seen))
                if minibatch % int(len(y)/(minibatch_size * n_times_predict_in_epoch)) == 0:
                    if X_valid is not None and y_valid is not None:
                        accuracy = self.calculate_accuracy(X_valid, y_valid, w2i, model, model_params)
                        self.logger.info("accuracy = {}".format(accuracy))

            # trainer.update_epoch() # this is depricated
            # total_loss /= len(y) # should this be done?
            self.logger.info('Epoch {}/{}, total_loss/examples_seen = {}'.format(epoch + 1, n_epochs,
                                                                                 total_loss / examples_seen))

        self.logger.info("Done training")

    def _predict_for_single_example(self, x, w2i, model_params):
        self.logger.debug("Starting to predict")

        # Renew the computational graph
        dy.renew_cg()

        # dynet expression for prob
        prob_of_each_word_at_every_pos = self._get_prob_of_each_word_at_every_pos(x, w2i, model_params)

        candidate_probs = {}
        candidates = x["candidates"]
        context = x["context"]
        for candidate in candidates:
            can_prob_exp = dy.esum([prob for prob, word in zip(prob_of_each_word_at_every_pos, context) if word == candidate])
            candidate_probs = {candidate: can_prob_exp.value()}

        # return the candidate with maximum prob
        return max(candidate_probs.iteritems(), key=operator.itemgetter(1))[0]

    def predict(self, X, w2i, model, model_params):
        self.logger.info("Starting to predict")

        answers = []
        for x in X:
            answers.append(self._predict_for_single_example(x, w2i, model_params))

        return answers

    def calculate_accuracy(self, X, y, w2i, model, model_params):
        self.logger.info("Starting to calculate accuracy")

        if len(y) is 0:
            self.logger.error("len(y) = {}".format(0))
            raise ValueError

        if len(y) != len(X):
            self.logger.error("len(y) = {} and len(X) = {} dont match.".format(len(y), len(X)))
            raise ValueError

        predicted_answers = self.predict(X, w2i, model, model_params)
        for i in range(5):
            print("predicted_answers = {}".format(predicted_answers[i]))
            print("candidates = {}".format(X[i]["candidates"]))

        accuracy = sum(1 for a, b in zip(predicted_answers, y) if a == b) / float(len(y))
        return accuracy







