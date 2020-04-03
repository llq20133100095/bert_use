import tensorflow as tf
import tensorflow_hub as hub
from hparame import Hparame
import bert
from prepro import process_data
from datetime import datetime
from bert import run_classifier
from bert import modeling
import logging
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score


logging.getLogger().setLevel(logging.INFO)
hparame = Hparame()
parser = hparame.parser
hp = parser.parse_args()
set_training = True


def create_model(bert_config, is_training, is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels, use_one_hot_embeddings):
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
    with tf.variable_scope("softmax_llq", reuse=tf.AUTO_REUSE):
        output_weights = tf.get_variable("output_weights", [num_labels, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):

        # Dropout helps prevent overfitting
        output_layer = tf.layers.dropout(output_layer, rate=0.1, training=is_training)

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


def train_eval_model(features, bert_config, num_labels, learning_rate, num_train_steps, num_warmup_steps, use_one_hot_embeddings,
          is_training, is_predicting):

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    (loss, predicted_labels, log_probs) = create_model(
        bert_config, is_training, is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels,
        use_one_hot_embeddings)

    train_op = bert.optimization.create_optimizer(
        loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

    global_step = tf.train.get_or_create_global_step()
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("global_step", global_step)

    accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
    eval_metrics = {
        "eval_accuracy": accuracy
    }

    tf.summary.scalar("eval_accuracy", accuracy[1])

    summaries = tf.summary.merge_all()
    return loss, train_op, global_step, summaries, eval_metrics, label_ids, predicted_labels


if __name__ == "__main__":
    label_list = [int(i) for i in hp.label_list.split(",")]
    train_features, eval_features = process_data(hp)

    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_features) / hp.BATCH_SIZE * hp.NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * hp.WARMUP_PROPORTION)
    num_eval_batches = len(eval_features) // hp.BATCH_SIZE + int(len(eval_features) % hp.BATCH_SIZE != 0)

    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = run_classifier.input_fn_builder(features=train_features, seq_length=hp.MAX_SEQ_LENGTH, is_training=True,
        drop_remainder=False)

    eval_input_fn = run_classifier.input_fn_builder(features=eval_features, seq_length=hp.MAX_SEQ_LENGTH, is_training=False,
        drop_remainder=False)

    train_batches = train_input_fn(params={"batch_size": hp.BATCH_SIZE})
    eval_batches = eval_input_fn(params={"batch_size": hp.BATCH_SIZE})
    eval_batches = eval_batches.repeat()

    # create a iterator of the correct shape and type
    iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
    features_input = iter.get_next()

    train_init_op = iter.make_initializer(train_batches)
    eval_init_op = iter.make_initializer(eval_batches)

    logging.info("# Load model")
    bert_config = modeling.BertConfig.from_json_file(hp.BERT_CONFIG)
    loss, train_op, global_step, train_summaries, eval_metrics_output, eval_label_ids, eval_predicted_labels = \
        train_eval_model(features=features_input, bert_config=bert_config, num_labels=len(label_list),
                         learning_rate=hp.LEARNING_RATE, num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps,
                         use_one_hot_embeddings=False, is_training=set_training, is_predicting=False)

    logging.info("# Session")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(hp.OUTPUT_DIR)
        if ckpt is None:
            logging.info("Initializing from scratch")
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        else:
            saver.restore(sess, ckpt)

        summary_writer = tf.summary.FileWriter(hp.OUTPUT_DIR, sess.graph)

        sess.run(train_init_op)
        _gs = sess.run(global_step)
        bleu_score = []

        for i in tqdm(range(_gs, num_train_steps+1)):
            _, _gs, _summary, _loss = sess.run([train_op, global_step, train_summaries, loss])
            summary_writer.add_summary(_summary, _gs)

            if _gs and _gs % 500 == 0:
                logging.info("# Loss")
                logging.info(_loss)

                logging.info("# save models")
                ckpt_name = os.path.join(hp.OUTPUT_DIR, hp.model_output)
                saver.save(sess, ckpt_name, global_step=_gs)

                logging.info("# test evaluation")
                set_training = False
                sess.run([eval_init_op])
                # summary_writer.add_summary(_eval_summaries, _gs)

                label_id_list = []
                predicted_label_list = []
                for _ in range(num_eval_batches):
                    _eval_label_ids, _eval_predicted_labels = sess.run([eval_label_ids, eval_predicted_labels])
                    label_id_list.extend(_eval_label_ids.tolist())
                    predicted_label_list.extend(_eval_predicted_labels.tolist())
                logging.info("eval nums %d " % len(predicted_label_list))
                logging.info("accuracy: %f" % accuracy_score(label_id_list, predicted_label_list))

                logging.info("# fall back to train mode")
                sess.run(train_init_op)
                set_training = True

        logging.info("# test evaluation")
        sess.run([eval_init_op])
        label_id_list = []
        predicted_label_list = []
        for _ in range(num_eval_batches):
            _eval_label_ids, _eval_predicted_labels = sess.run([eval_label_ids, eval_predicted_labels])
            label_id_list.extend(_eval_label_ids.tolist())
            predicted_label_list.extend(_eval_predicted_labels.tolist())
        logging.info("eval nums %d " % len(predicted_label_list))
        logging.info("final accuracy: %f" % accuracy_score(label_id_list, predicted_label_list))
