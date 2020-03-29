import tensorflow as tf
import tensorflow_hub as hub
from hparame import Hparame
import bert
from prepro import process_data
from datetime import datetime
from bert import run_classifier
from bert import modeling

hparame = Hparame()
parser = hparame.parser
hp = parser.parse_args()


def create_model(bert_config, is_training, is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels, use_one_hot_embeddings):
  # """Creates a classification model."""
  #
  # # bert_module = hub.Module(BERT_MODEL_HUB, trainable=True)
  # # bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
  # # bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)

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
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable("output_weights", [num_labels, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):

    # Dropout helps prevent overfitting
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Convert labels into one-hot encoding
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return (predicted_labels, log_probs)

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(bert_config, num_labels, learning_rate, num_train_steps, num_warmup_steps, use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:

            (loss, predicted_labels, log_probs) = create_model(
                bert_config, is_training, is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels, use_one_hot_embeddings)

            train_op = bert.optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            # Calculate evaluation metrics.
            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                f1_score = tf.contrib.metrics.f1_score(label_ids, predicted_labels)
                auc = tf.metrics.auc(label_ids, predicted_labels)
                recall = tf.metrics.recall(label_ids, predicted_labels)
                precision = tf.metrics.precision(label_ids, predicted_labels)
                true_pos = tf.metrics.true_positives(label_ids, predicted_labels)
                true_neg = tf.metrics.true_negatives(label_ids, predicted_labels)
                false_pos = tf.metrics.false_positives(label_ids, predicted_labels)
                false_neg = tf.metrics.false_negatives(label_ids, predicted_labels)
                return {
                    "eval_accuracy": accuracy,
                    "f1_score": f1_score,
                    "auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "true_positives": true_pos,
                    "true_negatives": true_neg,
                    "false_positives": false_pos,
                    "false_negatives": false_neg
                }

            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, log_probs) = create_model(
                bert_config, is_training, is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels, use_one_hot_embeddings)

            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn


if __name__ == "__main__":
    label_list = [int(i) for i in hp.label_list.split(",")]
    train_features, test_features = process_data(hp)

    """ Start train """
    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_features) / hp.BATCH_SIZE * hp.NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * hp.WARMUP_PROPORTION)

    # Specify output directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(model_dir=hp.OUTPUT_DIR, save_summary_steps=hp.SAVE_SUMMARY_STEPS, save_checkpoints_steps=hp.SAVE_CHECKPOINTS_STEPS)
    bert_config = modeling.BertConfig.from_json_file(hp.BERT_CONFIG)
    model_fn = model_fn_builder(bert_config=bert_config, num_labels=len(label_list), learning_rate=hp.LEARNING_RATE, num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps, use_one_hot_embeddings=False)

    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params={"batch_size": hp.BATCH_SIZE})

    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = run_classifier.input_fn_builder(features=train_features, seq_length=hp.MAX_SEQ_LENGTH, is_training=True,
        drop_remainder=False)

    current_time = datetime.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("Training took time ", datetime.now() - current_time)

    """ test prediction """
    test_input_fn = run_classifier.input_fn_builder(features=test_features, seq_length=hp.MAX_SEQ_LENGTH, is_training=False,
        drop_remainder=False)
    print(estimator.evaluate(input_fn=test_input_fn, steps=None))
