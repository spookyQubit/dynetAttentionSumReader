import dynet as dy
import os
import pickle
from ASReaderTrainer import ASReaderTrainer


class ASReader(object):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 gru_layers,
                 gru_input_dim,
                 gru_hidden_dim,
                 lookup_init_scale=1.0,
                 logger=None):

        self.logger = logger
        self.model_args = {"vocab_size": vocab_size,
                           "embedding_dim": embedding_dim,
                           "gru_layers": gru_layers,
                           "gru_input_dim": gru_input_dim,
                           "gru_hidden_dim": gru_hidden_dim,
                           "lookup_init_scale": lookup_init_scale}

        self.as_reader_trainer = ASReaderTrainer(logger)

        assert (self.model_args["gru_input_dim"] == self.model_args["embedding_dim"])

        self.model = None
        self.model_parameters = None

    def create_model(self):
        self.model, self.model_parameters = self._create_model()

    def load_model(self, model_save_file, model_args_save_file):
        if not os.path.exists(model_save_file):
            self.logger.error("{} does not exist to load model".format(model_save_file))
            raise ValueError

        self.logger.info("Loading model args from file: {}".format(model_args_save_file))
        # Load the hyper-parameter arguments needed to builds the model
        with open(model_args_save_file, "r") as f:
            self.model_args = pickle.load(f)

        self.logger.info("Loading model from file: {}".format(model_save_file))
        # Call the model parameters in the same order
        # which was used when creating the saved model in file_path
        self.model, self.model_parameters = self._create_model()
        self.model.populate(model_save_file)

        self.logger.info("Done loading model")

    def save_model(self, model_save_file, model_args_save_file):

        with open(model_save_file, "w+"):
            # Creating the file if it does not exist and clearing it if it doe exist
            self.logger.debug("Created/Emptied file {} to save file".format(model_save_file))

        if self.model is None:
            self.logger.error("model is already none")
            raise ValueError

        self.logger.info("Saving model in file: {}".format(model_save_file))
        self.model.save(model_save_file)

        self.logger.info("Saving model args in file: {}".format(model_args_save_file))
        with open(model_args_save_file, "w+") as f:
            pickle.dump(self.model_args, f)
        self.logger.info("Done saving model and model args")

    def _create_model(self):
        self.logger.info('Creating the model...')

        model = dy.Model()

        # context gru encoders
        c_fwdRnn = dy.GRUBuilder(self.model_args["gru_layers"],
                                 self.model_args["gru_input_dim"],
                                 self.model_args["gru_hidden_dim"],
                                 model)
        c_bwdRnn = dy.GRUBuilder(self.model_args["gru_layers"],
                                 self.model_args["gru_input_dim"],
                                 self.model_args["gru_hidden_dim"],
                                 model)

        # question gru encoders
        q_fwdRnn = dy.GRUBuilder(self.model_args["gru_layers"],
                                 self.model_args["gru_input_dim"],
                                 self.model_args["gru_hidden_dim"],
                                 model)
        q_bwdRnn = dy.GRUBuilder(self.model_args["gru_layers"],
                                 self.model_args["gru_input_dim"],
                                 self.model_args["gru_hidden_dim"],
                                 model)

        # embedding parameter
        lookup_params = model.add_lookup_parameters((self.model_args["vocab_size"],
                                                     self.model_args["gru_input_dim"]),
                                                    dy.UniformInitializer(self.model_args["lookup_init_scale"]))

        self.logger.info('Done creating the model')

        model_parameters = {"c_fwdRnn": c_fwdRnn,
                            "c_bwdRnn": c_bwdRnn,
                            "q_fwdRnn": q_fwdRnn,
                            "q_bwdRnn": q_bwdRnn,
                            "lookup_params": lookup_params}
        return model, model_parameters

    def fit(self,
            X, y, w2i,
            gradient_clipping_threshold,
            initial_learning_rate,
            n_epochs,
            minibatch_size,
            X_valid=None,
            y_valid=None,
            n_times_predict_in_epoch=3):

        self.as_reader_trainer.train(X, y, w2i,
                                     self.model, self.model_parameters,
                                     gradient_clipping_threshold,
                                     initial_learning_rate,
                                     n_epochs,
                                     minibatch_size,
                                     X_valid, y_valid)

    def get_accuracy(self, X, y, w2i):
        self.logger.info("Calculating accuracy with {} data points".format(len(y)))
        accuracy = self.as_reader_trainer.calculate_accuracy(X, y, w2i, self.model, self.model_parameters)
        self.logger.info("accuracy = {}".format(accuracy))
        return accuracy
