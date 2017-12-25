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
valid_file = "cbtest_NE_valid.txt"
test_file = "cbtest_NE_test.txt"

generated_data_dir = "generated_data"
vocab_file = "as_reader_vocab.txt")
w2i_file = "as_reader_w2i.txt"


# Model parameters
EMB_DIM = 128
GRU_LAYERS = 1
GRU_INPUT_DIM = EMB_DIM
GRU_HIDDEN_DIM = 128

# training parameters
N_EPOCHS = 10
ADAM_ALPHA = 0.01
MINIBATCH_SIZE = 16

"""

def get_logger():
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d-%m-%Y:%H:%M:%S',
                        filename='as_reader.log',
                        level=logging.INFO)
    logger = logging.getLogger('data_utils')
    return logger


def get_file_locations():

    # change this function to read the config file
    cbtdata_dir = "CBTest/data"
    train_file = "cbtest_NE_train.txt"
    valid_file = "cbtest_NE_valid.txt"
    test_file = "cbtest_NE_test.txt"

    train_files = [train_file]
    valid_files = [valid_file]
    test_files = [test_file]

    train_files = [os.path.join(cbtdata_dir, f) for f in train_files]
    valid_files = [os.path.join(cbtdata_dir, f) for f in valid_files]
    test_files = [os.path.join(cbtdata_dir, f) for f in test_files]

    generated_data_dir = "generated_data"
    vocab_file = os.path.join(generated_data_dir, "as_reader_vocab.txt")
    w2i_file = os.path.join(generated_data_dir, "as_reader_w2i.txt")

    return train_files, valid_files, test_files, vocab_file, w2i_file


def main():

    logger = get_logger()

    # get files
    train_files, valid_files, test_files, vocab_file, w2i_file = get_file_locations()

    cbt_data = CBTData(vocab_file=vocab_file,
                       w2i_file=w2i_file,
                       logger=logger,
                       max_data_points=1000)

    # Create vocabulary
    logger.info("Creating vocabulary")
    cbt_data.build_new_vocabulary_and_save(train_files, vocab_file)
    logger.info("Done creating vocabulary")

    # Create w2i
    logger.info("Creating w2i file")
    cbt_data.build_new_w2i_from_existing_vocab_and_save(w2i_file)
    logger.info("Done creating w2i file")

    # get training data
    logger.info("Getting training data")
    X_train, y_train = cbt_data.get_data(train_files)
    logger.info("Number of training data points = {}".format(len(y_train)))
    logger.info("Done getting training data")

    # Create ASReader instance
    EMB_DIM = 128
    GRU_LAYERS = 1
    GRU_INPUT_DIM = EMB_DIM
    GRU_HIDDEN_DIM = 128
    ADAM_ALPHA = 0.01
    MINIBATCH_SIZE = 16
    N_EPOCHS = 2
    as_reader = ASReader(vocab_size=len(cbt_data.get_vocab()),
                         embedding_dim=EMB_DIM,
                         gru_layers=GRU_LAYERS,
                         gru_input_dim=GRU_INPUT_DIM,
                         gru_hidden_dim=GRU_HIDDEN_DIM,
                         adam_alpha=ADAM_ALPHA,
                         minibatch_size=MINIBATCH_SIZE,
                         n_epochs=N_EPOCHS,
                         logger=logger,
                         w2i=cbt_data.get_w2i())

    # fit the model
    as_reader.fit(X_train, y_train)


if __name__ == "__main__":
    main()