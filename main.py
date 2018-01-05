from data_utils import CBTData
from ASReaderTrainer import ASReaderTrainer
from ASReaderConfig import ASReaderConfig
import os
import logging


def get_logger(cfg):
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d-%m-%Y:%H:%M:%S',
                        filename=cfg.get_log_file(),
                        level=logging.INFO)
    logger = logging.getLogger(cfg.get_logger())
    return logger


def get_file_locations(cfg):

    # change this function to read the config file
    cbtdata_dir = cfg.get_cbt_data_dir()
    train_file = cfg.get_cbtNE_train_file()
    valid_file = cfg.get_cbtNE_valid_file()
    test_file = cfg.get_cbtNE_test_file()

    train_files = [train_file]
    valid_files = [valid_file]
    test_files = [test_file]

    train_files = [os.path.join(cbtdata_dir, f) for f in train_files]
    valid_files = [os.path.join(cbtdata_dir, f) for f in valid_files]
    test_files = [os.path.join(cbtdata_dir, f) for f in test_files]

    generated_data_dir = cfg.get_generated_data_dir()
    vocab_file = os.path.join(generated_data_dir, cfg.get_vocab_file())
    w2i_file = os.path.join(generated_data_dir, cfg.get_w2i_file())
    train_save_file = os.path.join(generated_data_dir, cfg.get_train_save_file())
    valid_save_file = os.path.join(generated_data_dir, cfg.get_valid_save_file())
    test_save_file = os.path.join(generated_data_dir, cfg.get_test_save_file())
    model_save_file = os.path.join(generated_data_dir, cfg.get_model_save_file())
    model_args_save_file = os.path.join(generated_data_dir, cfg.get_model_args_save_file())

    return train_files, train_save_file, valid_files, valid_save_file, test_files, test_save_file, vocab_file, w2i_file, model_save_file, model_args_save_file


def get_cbt_data(vocab_file, w2i_file, cfg, logger):

    cbt_data = CBTData(vocab_file=vocab_file,
                       w2i_file=w2i_file,
                       logger=logger,
                       max_data_points=cfg.get_max_data_points())
    return cbt_data


def get_as_reader_trainer(cbt_data, cfg, logger):
    # Create ASReader instance
    as_reader_trainer = ASReaderTrainer(vocab_size=len(cbt_data.get_vocab()),
                                        embedding_dim=cfg.get_emb_dim(),
                                        gru_layers=cfg.get_gru_layers(),
                                        gru_input_dim=cfg.get_gru_input_dim(),
                                        gru_hidden_dim=cfg.get_gru_hidden_dim(),
                                        lookup_init_scale=cfg.get_lookup_init_scale(),
                                        number_of_unks=cfg.get_num_of_unk(),
                                        logger=logger)
    return as_reader_trainer


def setup_training(cfg, logger):
    # get files
    train_files, train_save_file, \
        valid_files, valid_save_file, \
        test_files, test_save_file, \
        vocab_file, \
        w2i_file, \
        model_save_file, \
        model_args_save_file = get_file_locations(cfg)

    cbt_data = get_cbt_data(vocab_file, w2i_file, cfg, logger)

    X_train = []
    y_train = []
    X_valid = []
    y_valid = []
    X_test = []
    y_test = []

    if cfg.get_should_load_saved_data():
        logger.info("Loading existing vocab and w2i")
        cbt_data.load_vocabulary(vocab_file)
        cbt_data.load_w2i(w2i_file)
        X_train, y_train = cbt_data.load_data(train_save_file)
        X_valid, y_valid = cbt_data.load_data(valid_save_file)
        X_test, y_test = cbt_data.load_data(test_save_file)
    else:
        logger.info("Creating new vocab and w2i")

        # Note that in paper, the vocab is created from train + valid + test files
        # Here we create only from train (and later from train + valid)
        # Generate vocab
        cbt_data.build_new_vocabulary_and_save(train_files,
                                               vocab_file,
                                               keep_top_vocab_percent=cfg.get_keep_top_vocab_percentage())

        # Generate w2i
        cbt_data.build_new_w2i_from_existing_vocab_and_save(w2i_file)

        # Get training data
        X_train, y_train = cbt_data.get_data_and_save(train_files, train_save_file)
        X_valid, y_valid = cbt_data.get_data_and_save(valid_files, valid_save_file)
        X_test, y_test = cbt_data.get_data_and_save(test_files, test_save_file)

    logger.info("Vocab size = {}".format(len(cbt_data.get_vocab())))
    logger.info("Number of training data points = {}".format(len(y_train)))
    logger.info("Number of validation data points = {}".format(len(y_valid)))
    logger.info("Number of testing data points = {}".format(len(y_test)))

    # get the attention sum reader instance
    as_reader_trainer = get_as_reader_trainer(cbt_data, cfg, logger)

    if cfg.get_should_load_saved_model():
        as_reader_trainer.as_reader_model.load_model(model_save_file, model_args_save_file)
    else:
        as_reader_trainer.as_reader_model.create_model()

    # fit the model
    as_reader_trainer.train(X=X_train, y=y_train, w2i=cbt_data.get_w2i(),
                            gradient_clipping_threshold=cfg.get_gradient_clipping_thresh(),
                            initial_learning_rate=cfg.get_adam_alpha(),
                            n_epochs=cfg.get_n_epochs(),
                            minibatch_size=cfg.get_minibatch_size(),
                            X_valid=X_valid, y_valid=y_valid,
                            n_times_predict_in_epoch=cfg.get_n_times_predict_in_epoch(),
                            should_save_model_while_training=cfg.get_should_save_model_while_training(),
                            model_save_file=model_save_file,
                            model_args_save_file=model_args_save_file)

    test_accuracy = as_reader_trainer.calculate_accuracy(X_test, y_test, cbt_data.get_w2i())
    logger.info("test_accuracy = {}".format(test_accuracy))

    as_reader_trainer.as_reader_model.save_model(model_save_file, model_args_save_file)


def setup_testing(cfg, logger):
    logger.info("testing")

    train_files, train_save_file, \
    valid_files, valid_save_file, \
    test_files, test_save_file, \
    vocab_file, \
    w2i_file, \
    model_save_file, \
    model_args_save_file = get_file_locations(cfg)

    cbt_data = get_cbt_data(vocab_file, w2i_file, cfg, logger)

    # We are assuming here that the test data is already saved
    X_test, y_test = cbt_data.load_data(test_save_file)
    logger.info("Number of testing data points = {}".format(len(y_test)))

    # get the attention sum reader instance
    as_reader_trainer = get_as_reader_trainer(cbt_data, cfg, logger)

    # We are assuming that the model is already saved
    as_reader_trainer.as_reader_model.load_model(model_save_file, model_args_save_file)

    test_accuracy = as_reader_trainer.calculate_accuracy(X_test, y_test, cbt_data.get_w2i())
    logger.info("test_accuracy = {}".format(test_accuracy))


def main():
    cfg = ASReaderConfig()

    logger = get_logger(cfg)
    logger.info("in main")

    if cfg.get_execution_mode() == "training":
        setup_training(cfg, logger)
    elif cfg.get_execution_mode() == "testing":
        setup_testing(cfg, logger)
    else:
        logger.error("Unsupported mode")

if __name__ == "__main__":
    main()
