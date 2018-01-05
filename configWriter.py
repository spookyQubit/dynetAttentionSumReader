import ConfigParser


def set_section_logging(cfg):
    # Section: Logging
    cfg.add_section('Logging')
    cfg.set('Logging', 'log_level', 'info')
    cfg.set('Logging', 'log_file', '/home/shantanu/PycharmProjects/attentionSum/as_reader.log')
    cfg.set('Logging', 'logger', 'ASReader')


def set_section_data(cfg):
    # Section data
    cfg.add_section('Data')
    cfg.set('Data', 'cbt_data_dir', '/home/shantanu/PycharmProjects/attentionSum/CBTest/data')
    cfg.set('Data', 'cbtNE_train_file', 'cbtest_NE_train.txt')
    cfg.set('Data', 'cbtNE_valid_file', 'cbtest_NE_valid_2000ex.txt')
    cfg.set('Data', 'cbtNE_test_file', 'cbtest_NE_test_2500ex.txt')

    cfg.set('Data', 'generated_data_dir', '/home/shantanu/PycharmProjects/attentionSum/generated_data')
    cfg.set('Data', 'vocab_file', 'as_reader_vocab.txt')
    cfg.set('Data', 'w2i_file', 'as_reader_w2i.txt')
    cfg.set('Data', 'train_save_file', 'cbtest_NE_train_save.txt')
    cfg.set('Data', 'valid_save_file', 'cbtest_NE_valid_save.txt')
    cfg.set('Data', 'test_save_file', 'cbtest_NE_test_save.txt')
    cfg.set('Data', 'model_save_file', 'model_save.txt')
    cfg.set('Data', 'model_args_save_file', 'model_args_save.txt')

    cfg.set('Data', 'keep_top_vocab_percentage', '90.0')
    cfg.set('Data', 'max_data_points')  # Default is none
    cfg.set('Data', 'should_load_saved_data', 'false')


def set_section_execution_mode(cfg):
    # Section mode
    cfg.add_section('ExecutionMode')
    cfg.set('ExecutionMode', 'mode', 'training')


def set_section_execution_model_params(cfg):
    # Section model params
    cfg.add_section('ModelParameters')
    cfg.set('ModelParameters', 'emb_dim', '128')
    cfg.set('ModelParameters', 'gru_layers', '1')
    cfg.set('ModelParameters', 'gru_input_dim', '128')
    cfg.set('ModelParameters', 'gru_hidden_dim', '128')
    cfg.set('ModelParameters', 'num_of_unk', '10')
    cfg.set('ModelParameters', 'should_load_saved_model', 'false')


def set_section_training_params(cfg):
    cfg.add_section('TrainingParameters')
    cfg.set('TrainingParameters', 'adam_alpha', '0.001')
    cfg.set('TrainingParameters', 'minibatch_size', '16')
    cfg.set('TrainingParameters', 'n_epochs', '2')
    cfg.set('TrainingParameters', 'gradient_clipping_thresh', '10.0')
    cfg.set('TrainingParameters', 'lookup_init_scale', '1.0')
    cfg.set('TrainingParameters', 'n_times_predict_in_epoch', '2')
    cfg.set('TrainingParameters', 'should_save_model_while_training', 'true')


def main():
    cfg = ConfigParser.RawConfigParser()

    set_section_logging(cfg)
    set_section_data(cfg)
    set_section_execution_mode(cfg)
    set_section_execution_model_params(cfg)
    set_section_training_params(cfg)

    with open('/home/shantanu/PycharmProjects/attentionSum/ASReader.cfg', 'wb') as configfile:
        cfg.write(configfile)


if __name__ == "__main__":
    main()
