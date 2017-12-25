from data_utils import CBTData
import os
import logging


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
                       logger=logger)

    logger.info("Creating vocabulary")
    cbt_data.build_new_vocabulary_and_save(train_files, vocab_file)
    logger.info("Done creating vocabulary")

    logger.info("Creating w2i file")
    cbt_data.build_new_w2i_from_existing_vocab_and_save(w2i_file)
    logger.info("Done creating w2i file")

    # get training data
    logger.info("Getting training data")
    X_train, y_train = cbt_data.get_data(train_files)
    logger.info("Number of training data points = {}".format(len(y_train)))
    logger.info("Done getting training data")


if __name__ == "__main__":
    main()