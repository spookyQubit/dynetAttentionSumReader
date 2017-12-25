import dynet as dy
from dynet import GRUBuilder
import logging
import math

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
        self.logger = logger

        assert(self.gru_input_dim == self.embedding_dim)

        self.model, self.f_builder, self.b_builder, self.model_parameters = self._create_model()

    def fit(self, X, y):
        self.train(X, y, self.w2i)

    def train(self, X, y, w2i):
        self.logger.info("Starting to train")

        trainer = dy.AdamTrainer(self.model, self.adam_alpha)
        n_minibatches = int(math.ceil(len(y) / self.minibatch_size))

        for epoch in range(self.n_epochs):


        self.logger.info("Starting to train")

    def predict(self):
        pass

    def load_model(self):
        pass

    def save_model(self):
        pass

    def _create_model(self):
        self.logger.info('Creating the model...')
        model = dy.Model()
        f_builder = dy.GRUBuilder(self.gru_layers,
                                  self.gru_input_dim,
                                  self.gru_hidden_dim,
                                  model)
        b_builder = dy.GRUBuilder(self.gru_layers,
                                  self.gru_input_dim,
                                  self.gru_hidden_dim,
                                  model)
        model_parameters = {}
        model_parameters["lookup"] = model.add_lookup_parameters((self.vocab_size,
                                                                  self.gru_input_dim))
        self.logger.info('Done creating the model')
        return model, f_builder, b_builder, model_parameters
