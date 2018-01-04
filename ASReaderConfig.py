import ConfigParser


class ASReaderConfig(object):
    def __init__(self):
        self.cfg = ConfigParser.RawConfigParser()
        self.cfg.read('/home/shantanu/PycharmProjects/attentionSum/ASReader.cfg')

    def get_log_level(self):
        return self.cfg.get('Logging', 'log_level')

    def get_log_file(self):
        return self.cfg.get('Logging', 'log_file')

    def get_logger(self):
        return self.cfg.get('Logging', 'logger')

    def get_cbt_data_dir(self):
        return self.cfg.get('Data', 'cbt_data_dir')

    def get_cbtNE_train_file(self):
        return self.cfg.get('Data', 'cbtNE_train_file')

    def get_cbtNE_valid_file(self):
        return self.cfg.get('Data', 'cbtNE_valid_file')

    def get_cbtNE_test_file(self):
        return self.cfg.get('Data', 'cbtNE_test_file')

    def get_generated_data_dir(self):
        return self.cfg.get('Data', 'generated_data_dir')

    def get_vocab_file(self):
        return self.cfg.get('Data', 'vocab_file')

    def get_w2i_file(self):
        return self.cfg.get('Data', 'w2i_file')

    def get_train_save_file(self):
        return self.cfg.get('Data', 'train_save_file')

    def get_valid_save_file(self):
        return self.cfg.get('Data', 'valid_save_file')

    def get_test_save_file(self):
        return self.cfg.get('Data', 'test_save_file')

    def get_model_save_file(self):
        return self.cfg.get('Data', 'model_save_file')

    def get_model_args_save_file(self):
        return self.cfg.get('Data', 'model_args_save_file')

    def get_keep_top_vocab_percentage(self):
        return self.cfg.getfloat('Data', 'keep_top_vocab_percentage')

    def get_max_data_points(self):
        if self.cfg.get('Data', 'max_data_points') == "None":
            return None
        else:
            return self.cfg.getint('Data', 'max_data_points')

    def get_should_load_saved_data(self):
        return self.cfg.getboolean('Data', 'should_load_saved_data')

    def get_execution_mode(self):
        return self.cfg.get('ExecutionMode', 'mode')

    def get_emb_dim(self):
        return self.cfg.getint('ModelParameters', 'emb_dim')

    def get_gru_layers(self):
        return self.cfg.getint('ModelParameters', 'gru_layers')

    def get_gru_input_dim(self):
        return self.cfg.getint('ModelParameters', 'gru_input_dim')

    def get_gru_hidden_dim(self):
        return self.cfg.getint('ModelParameters', 'gru_hidden_dim')

    def get_num_of_unk(self):
        return self.cfg.getint('ModelParameters', 'num_of_unk')

    def get_should_load_saved_model(self):
        return self.cfg.getboolean('ModelParameters', 'should_load_saved_model')

    def get_adam_alpha(self):
        return self.cfg.getfloat('TrainingParameters', 'adam_alpha')

    def get_minibatch_size(self):
        return self.cfg.getint('TrainingParameters', 'minibatch_size')

    def get_n_epochs(self):
        return self.cfg.getint('TrainingParameters', 'n_epochs')

    def get_gradient_clipping_thresh(self):
        return self.cfg.getfloat('TrainingParameters', 'gradient_clipping_thresh')

    def get_lookup_init_scale(self):
        return self.cfg.getfloat('TrainingParameters', 'lookup_init_scale')

    def get_n_times_predict_in_epoch(self):
        return self.cfg.getint('TrainingParameters', 'n_times_predict_in_epoch')

    def get_should_save_model_while_training(self):
        return self.cfg.getboolean('TrainingParameters', 'should_save_model_while_training')