from data_utils import CBTData
from as_reader import ASReader
import os
import logging

"""
To define in cfg

log:
LOG_LEVEL
log_file

Data:
cbtdata_dir = "CBTest/data"
train_file = "cbtest_NE_train.txt"
valid_file = "cbtest_NE_valid_200ex.txt"
test_file = "cbtest_NE_test_2500ex.txt"

generated_data_dir = "generated_data"
vocab_file = "as_reader_vocab.txt")
w2i_file = "as_reader_w2i.txt"
train_save_file = "cbtest_NE_train_save.txt"
valid_save_file = "cbtest_NE_valid_save.txt"
test_save_file = "cbtest_NE_test_save.txt"



max_data_points = 10000
SHOULD_CREATE_NEW_VOCAB = True
MODE = "training"/"testing"

# Model parameters
EMB_DIM = 128
GRU_LAYERS = 1
GRU_INPUT_DIM = EMB_DIM
GRU_HIDDEN_DIM = 128

# training parameters
ADAM_ALPHA = 0.001
MINIBATCH_SIZE = 16
N_EPOCHS = 2
GRADIENT_CLIPPING_THRESHOLD = 10.0
LOOKUP_INIT_SCALE = 1.0
"""


def get_logger():
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d-%m-%Y:%H:%M:%S',
                        filename='/home/shantanu/PycharmProjects/attentionSum/as_reader.log',
                        level=logging.INFO)
    logger = logging.getLogger('data_utils')
    return logger


def get_mode():
    MODE = "training"
    return MODE


def get_file_locations():

    # change this function to read the config file
    cbtdata_dir = "/home/shantanu/PycharmProjects/attentionSum/CBTest/data"
    train_file = "cbtest_NE_train.txt"
    valid_file = "cbtest_NE_valid_2000ex.txt"
    test_file = "cbtest_NE_test_2500ex.txt"

    train_files = [train_file]
    valid_files = [valid_file]
    test_files = [test_file]

    train_files = [os.path.join(cbtdata_dir, f) for f in train_files]
    valid_files = [os.path.join(cbtdata_dir, f) for f in valid_files]
    test_files = [os.path.join(cbtdata_dir, f) for f in test_files]

    generated_data_dir = "/home/shantanu/PycharmProjects/attentionSum/generated_data"
    vocab_file = os.path.join(generated_data_dir, "as_reader_vocab.txt")
    w2i_file = os.path.join(generated_data_dir, "as_reader_w2i.txt")
    train_save_file = os.path.join(generated_data_dir, "cbtest_NE_train_save.txt")
    valid_save_file = os.path.join(generated_data_dir, "cbtest_NE_valid_save.txt")
    test_save_file = os.path.join(generated_data_dir, "cbtest_NE_test_save.txt")
    model_save_file = os.path.join(generated_data_dir, "model_save.txt")
    model_args_save_file = os.path.join(generated_data_dir, "model_args_save.txt")

    return train_files, train_save_file, valid_files, valid_save_file, test_files, test_save_file, vocab_file, w2i_file, model_save_file, model_args_save_file


def get_cbt_data(vocab_file, w2i_file, logger):
    cbt_data = CBTData(vocab_file=vocab_file,
                       w2i_file=w2i_file,
                       logger=logger)
    return cbt_data


def get_training_args():
    training_args = {}
    training_args["ADAM_ALPHA"] = 0.001
    training_args["MINIBATCH_SIZE"] = 16
    training_args["N_EPOCHS"] = 10
    training_args["GRADIENT_CLIPPING_THRESHOLD"] = 10.0
    return training_args


def get_as_reader(cbt_data, logger):
    # Create ASReader instance
    EMB_DIM = 128
    GRU_LAYERS = 1
    GRU_INPUT_DIM = EMB_DIM
    GRU_HIDDEN_DIM = 128
    LOOKUP_INIT_SCALE = 1.0
    NUMBER_OF_UNKS = 1000
    as_reader = ASReader(vocab_size=len(cbt_data.get_vocab()),
                         embedding_dim=EMB_DIM,
                         gru_layers=GRU_LAYERS,
                         gru_input_dim=GRU_INPUT_DIM,
                         gru_hidden_dim=GRU_HIDDEN_DIM,
                         lookup_init_scale=LOOKUP_INIT_SCALE,
                         number_of_unks=NUMBER_OF_UNKS,
                         logger=logger)
    return as_reader


def should_load_saved_data():
    SHOULD_LOAD_SAVED_DATA = True
    return SHOULD_LOAD_SAVED_DATA


def should_load_a_saved_model():
    SHOULD_LOAD_A_SAVED_MODEL = False
    return SHOULD_LOAD_A_SAVED_MODEL


def setup_training(logger):
    # get files
    train_files, train_save_file, \
        valid_files, valid_save_file, \
        test_files, test_save_file, \
        vocab_file, \
        w2i_file, \
        model_save_file, \
        model_args_save_file = get_file_locations()

    cbt_data = get_cbt_data(vocab_file, w2i_file, logger)

    X_train = []
    y_train = []

    if should_load_saved_data():
        logger.info("Loading existing vocab and w2i")
        cbt_data.load_vocabulary(vocab_file)
        cbt_data.load_w2i(w2i_file)
        X_train, y_train = cbt_data.load_data(train_save_file)
        X_valid, y_valid = cbt_data.load_data(valid_save_file)
    else:
        logger.info("Creating new vocab and w2i")

        # Note that the vocab should be created from train + valid + test files
        # Generate vocab
        cbt_data.build_new_vocabulary_and_save(train_files + valid_files, vocab_file)

        # Generate w2i
        cbt_data.build_new_w2i_from_existing_vocab_and_save(w2i_file)

        # Get training data
        X_train, y_train = cbt_data.get_data_and_save(train_files, train_save_file)
        X_valid, y_valid = cbt_data.get_data_and_save(valid_files, valid_save_file)

    logger.info("Number of training data points = {}".format(len(y_train)))
    logger.info("Vocab size = {}".format(len(cbt_data.get_vocab())))

    # get the attention sum reader instance
    as_reader = get_as_reader(cbt_data, logger)

    if should_load_a_saved_model():
        as_reader.load_model(model_save_file, model_args_save_file)
    else:
        as_reader.create_model()

    # fit the model
    N_TIMES_PREDICT_IN_EPOCH = 2
    training_args = get_training_args()
    as_reader.fit(X_train, y_train, cbt_data.get_w2i(),
                  training_args["GRADIENT_CLIPPING_THRESHOLD"],
                  training_args["ADAM_ALPHA"],
                  training_args["N_EPOCHS"],
                  training_args["MINIBATCH_SIZE"],
                  X_valid, y_valid,
                  N_TIMES_PREDICT_IN_EPOCH)

    # accuracy = as_reader.get_accuracy(X_valid, y_valid, cbt_data.get_w2i())
    # logger.info("accuracy = {}".format(accuracy))

    as_reader.save_model(model_save_file, model_args_save_file)


def setup_testing(logger):
    logger.info("testing not yet implemented")
    pass


def main():

    logger = get_logger()
    logger.info("in main")

    mode = get_mode()
    if mode is "training":
        setup_training(logger)
    elif mode is "testing":
        setup_testing(logger)
    else:
        logger.error("Unsupported mode")

if __name__ == "__main__":
    main()