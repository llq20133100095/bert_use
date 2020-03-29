"""
@author: llq
@fucntion: output sentence embedddings
"""
import tensorflow as tf
import tensorflow_hub as hub
from hparame import Hparame
import bert
from prepro import process_data, create_tokenizer_from_hub_module
from datetime import datetime
from bert import run_classifier
from bert import modeling

hparame = Hparame()
parser = hparame.parser
hp = parser.parse_args()
label_list = [int(i) for i in hp.label_list.split(",")]


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, use_one_hot_embeddings, use_sentence):
    """Creates a classification model."""
    model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_outputs" for token-level output.
    if use_sentence:
        output_layer = model.get_pooled_output()
    else:
        output_layer = model.get_sequence_output()

    return output_layer


def model_fn_builder(bert_config, use_one_hot_embeddings, use_sentence):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        output_layer = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, use_one_hot_embeddings, use_sentence)

        predictions = {
            'sentence_embeddings': output_layer,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn


def getPrediction(in_sentences, estimator):
    labels = ["Negative", "Positive"]
    tokenizer = create_tokenizer_from_hub_module(hp)

    input_examples = [run_classifier.InputExample(guid="", text_a=x, text_b=None, label = 0) for x in in_sentences] # here, "" is just a dummy label
    input_features = run_classifier.convert_examples_to_features(input_examples, label_list, hp.MAX_SEQ_LENGTH, tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=hp.MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)

    predictions = estimator.predict(predict_input_fn)
    return [(sentence, prediction['sentence_embeddings'].shape) for sentence, prediction in zip(in_sentences, predictions)]


if __name__ == "__main__":
    # bulid estimator
    # Specify output directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(model_dir=hp.OUTPUT_DIR, save_summary_steps=hp.SAVE_SUMMARY_STEPS, save_checkpoints_steps=hp.SAVE_CHECKPOINTS_STEPS)
    bert_config = modeling.BertConfig.from_json_file(hp.BERT_CONFIG)
    model_fn = model_fn_builder(bert_config=bert_config, use_one_hot_embeddings=False, use_sentence=True)

    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params={"batch_size": hp.BATCH_SIZE})

    pred_sentences = [
        "That movie was absolutely awful",
        "The acting was a bit lacking",
        "The film was creative and surprising",
        "Absolutely fantastic!"
    ]

    print(getPrediction(pred_sentences, estimator))