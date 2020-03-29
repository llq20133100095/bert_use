from data_load import download_and_load_datasets
from hparame import Hparame
import bert
import tensorflow as tf
import tensorflow_hub as hub
from bert import tokenization
from bert import run_classifier


def create_tokenizer_from_hub_module(hp):
    """
    create tokenizer
    :return:
    """
    tokenization.validate_case_matches_checkpoint(True, hp.BERT_INIT_CHKPNT)

    return tokenization.FullTokenizer(vocab_file=hp.BERT_VOCAB, do_lower_case=True)


def process_data(hp):
    tokenizer = create_tokenizer_from_hub_module(hp)

    train, test = download_and_load_datasets()
    # train = train.sample(5000)
    # test = test.sample(5000)

    # Use the InputExample class from BERT's run_classifier code to create examples from the data
    train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                   # Globally unique ID for bookkeeping, unused in this example
                                                   text_a=x[hp.DATA_COLUMN],
                                                   text_b=None,
                                                   label=x[hp.LABEL_COLUMN]), axis=1)

    test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                   text_a=x[hp.DATA_COLUMN],
                                                   text_b=None,
                                                   label=x[hp.LABEL_COLUMN]), axis=1)

    # print(tokenizer.tokenize("This here's an example of using the BERT tokenizer"))

    # Convert our train and test features to InputFeatures that BERT understands.
    label_list = [int(i) for i in hp.label_list.split(",")]
    train_features = run_classifier.convert_examples_to_features(train_InputExamples, label_list, hp.MAX_SEQ_LENGTH,
                                                                      tokenizer)
    test_features = run_classifier.convert_examples_to_features(test_InputExamples, label_list, hp.MAX_SEQ_LENGTH,
                                                                     tokenizer)
    return train_features, test_features


if __name__ == "__main__":
    hparame = Hparame()
    parser = hparame.parser
    hp = parser.parse_args()

    train_features, test_features = process_data(hp)