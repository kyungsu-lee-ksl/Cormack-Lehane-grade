import cv2
import tensorflow as tf
import numpy as np
from enum import Enum
from util.model.Layer import conv_layer, batch_norm, max_pooling, avg_pooling
from typing import Tuple, List, Callable

from util.model.callback import CallBack
from util.model.dataset import Dataset


class Model:
    def __init__(self,
                 input_shape: tuple = (256, 256, 3),
                 annot_shape: tuple or None = None,
                 batch_size: int = 4,
                 num_classes: int = 2,
                 ):

        assert len(input_shape) == 3, f"input shape should be (?, ?, ?), but {input_shape}."
        assert annot_shape is None or len(annot_shape) == 3, f"input shape should be (?, ?, ?), but {input_shape}."

        self.__num_classes__ = num_classes
        self.__batchSize__ = batch_size
        self.__inputShape__ = input_shape
        self.__annotShape__ = (int(input_shape[0]), int(input_shape[1]), 1) if annot_shape is None else annot_shape

        self.model = None
        self.inputHolder = tf.placeholder(dtype=tf.float32, shape=(None, int(self.__inputShape__[0]), int(self.__inputShape__[1]), int(self.__inputShape__[2])))
        self.annotHolder = tf.placeholder(dtype=tf.int64, shape=(None,))

        self.logit1, self.logit2, self.cam = self.__build__()

    def conv(self, inputs, input_channel, output_channel, name='conv', activation=tf.nn.relu):
        layer = conv_layer(inputs, input_channel=input_channel, output_channel=output_channel, name='%s__conv' % name)
        layer = batch_norm(layer, output_channel, name='%s__bn' % name)
        layer = activation(layer)
        return layer

    def conv_and_pooling(self, inputs, output_channel, name='conv', activation=tf.nn.relu):
        layer = self.conv(inputs=inputs, input_channel=int(inputs.shape[-1]), output_channel=output_channel, name=f"{name}_01", activation=activation)
        layer = self.conv(inputs=layer, input_channel=output_channel, output_channel=output_channel, name=f"{name}_02", activation=activation)
        layer = self.conv(inputs=layer, input_channel=output_channel, output_channel=output_channel, name=f"{name}_03", activation=activation)
        layer = max_pooling(layer, name=f"{name}_max_pool")
        return layer

    def conv_only(self, inputs, output_channel, name='conv', activation=tf.nn.relu):
        layer = self.conv(inputs=inputs, input_channel=int(inputs.shape[-1]), output_channel=output_channel, name=f"{name}_01", activation=activation)
        layer = self.conv(inputs=layer, input_channel=output_channel, output_channel=output_channel, name=f"{name}_02", activation=activation)
        layer = self.conv(inputs=layer, input_channel=output_channel, output_channel=output_channel, name=f"{name}_03", activation=activation)
        return layer

    def pooling(self, layer, coef=2):
        return avg_pooling(layer, coef)

    def min_max_norm(self, tensor):
        return (tensor - tf.reduce_min(tensor)) / (tf.reduce_max(tensor) - tf.reduce_min(tensor))

    def sim_factor(self, tensor, cam, method: Callable = tf.multiply):
        return method(tensor, cam) / (method(tensor, tensor) + method(cam, cam) - method(tensor, cam))
        # return tf.multiply(tensor, cam) / (tf.multiply(tensor, tensor) + tf.multiply(cam, cam) - tf.multiply(tensor, cam))
        # return tf.matmul(tensor, cam) / (tf.matmul(tensor, tensor) + tf.matmul(cam, cam) - tf.matmul(tensor, cam))

    def __build__(self):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            # pipeline # 2 (lower)
            layer2_01 = self.conv_only(self.inputHolder, 64, name='layer2_01')
            layer2_02 = self.conv_only(layer2_01, 128, name='layer2_02')
            layer2_03 = self.conv_only(layer2_02, 256, name='layer2_03')
            layer2_04 = self.conv_only(layer2_03, 512, name='layer2_04')
            layer2_05 = self.conv_only(layer2_04, 1024, name='layer2_05')

            # pipeline # 1 (upper)
            layer1_01 = self.conv_only(self.inputHolder, 64, name='layer1_01') + layer2_01
            layer1_02 = self.conv_and_pooling(layer1_01, 128, name='layer1_02') + self.pooling(layer2_02, 2)
            layer1_03 = self.conv_and_pooling(layer1_02, 256, name='layer1_03') + self.pooling(layer2_03, 4)
            layer1_04 = self.conv_and_pooling(layer1_03, 512, name='layer1_04') + self.pooling(layer2_04, 8)
            layer1_05 = self.conv_and_pooling(layer1_04, 1024, name='layer1_05') + self.pooling(layer2_05, 16)

            # merging pipeline
            layer3 = tf.reduce_mean(layer1_05, axis=[1, 2])
            weights_01 = tf.get_variable(name='attention_weights_01', trainable=True, initializer=tf.truncated_normal(shape=(1024,), mean=0.0, stddev=1.0))
            weights_02 = tf.get_variable(name='attention_weights_02', trainable=True, initializer=tf.truncated_normal(shape=(1024,), mean=0.0, stddev=1.0))

            layer4_01 = tf.reduce_sum(layer3 * weights_01, axis=-1, keep_dims=True)
            layer4_02 = tf.reduce_sum(layer3 * weights_02, axis=-1, keep_dims=True)

            logit_01 = tf.concat([layer4_01, layer4_02], axis=-1)

            # CAM for positive prediction
            CAM = self.min_max_norm(tf.reduce_sum(weights_02 * tf.image.resize(layer1_05, size=[self.__inputShape__[0], self.__inputShape__[1]], method=tf.image.ResizeMethod.BILINEAR), axis=-1))

            # normalized feature maps
            n_layer2_01, n_layer2_02, n_layer2_03, n_layer2_04, n_layer2_05 = tuple(
                map(lambda x: self.min_max_norm(tf.reduce_mean(x, axis=-1)), [layer2_01, layer2_02, layer2_03, layer2_04, layer2_05]))

            # similarity factor
            sf_layer2_01, sf_layer2_02, sf_layer2_03, sf_layer2_04, sf_layer2_05 = tuple(
                map(lambda x: tf.expand_dims(self.min_max_norm(self.sim_factor(x, CAM)), axis=-1), [n_layer2_01, n_layer2_02, n_layer2_03, n_layer2_04, n_layer2_05]))

            # re_normalize
            new_layer2_01 = sf_layer2_01 * layer2_01
            new_layer2_02 = sf_layer2_02 * layer2_02 + self.conv_only(new_layer2_01, 128, name='layer2_02')
            new_layer2_03 = sf_layer2_03 * layer2_03 + self.conv_only(new_layer2_02, 256, name='layer2_03')
            new_layer2_04 = sf_layer2_04 * layer2_04 + self.conv_only(new_layer2_03, 512, name='layer2_04')
            new_layer2_05 = sf_layer2_05 * layer2_05 + self.conv_only(new_layer2_04, 1024, name='layer2_05')

            new_layer1_01 = self.conv_only(self.inputHolder, 64, name='layer1_01') + new_layer2_01
            new_layer1_02 = self.conv_and_pooling(new_layer1_01, 128, name='layer1_02') + self.pooling(new_layer2_02, 2)
            new_layer1_03 = self.conv_and_pooling(new_layer1_02, 256, name='layer1_03') + self.pooling(new_layer2_03, 4)
            new_layer1_04 = self.conv_and_pooling(new_layer1_03, 512, name='layer1_04') + self.pooling(new_layer2_04, 8)
            new_layer1_05 = self.conv_and_pooling(new_layer1_04, 1024, name='layer1_05') + self.pooling(new_layer2_05, 16)

            # final prediction
            layer5 = tf.reduce_mean(new_layer1_05, axis=[1, 2])
            logit2 = tf.layers.dense(layer5, 2, activation=None)

        return logit_01, logit2, CAM

    def __getComponents__(self):
        return self.inputHolder, self.annotHolder, self.logit1, self.logit2, self.cam

    def compile(self, optimizer=None, learning_rate: float = 1e-2, sess=None, load_path=None):
        _, annotHolder, logit1, logit2, cam = self.__getComponents__()

        loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(annotHolder, depth=self.__num_classes__), logits=logit1)
        self.loss1 = tf.reduce_mean(loss1)

        loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(annotHolder, depth=self.__num_classes__), logits=logit2)
        self.loss2 = tf.reduce_mean(loss2)

        if optimizer is None:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # manage trainable variables
        vars1, vars2 = list(), list()
        for v in tf.trainable_variables():
            if v.name.find('layer2') >= 0 or v.name.find('dense') >= 0:
                vars2.append(v)
            if not (v.name.find('dense') >= 0):
                vars1.append(v)

        # calculate gradients
        train_grads1 = optimizer.compute_gradients(loss=loss1, var_list=vars1)
        train_grads2 = optimizer.compute_gradients(loss=loss2, var_list=vars2)

        self.train_op1 = optimizer.apply_gradients(train_grads1)
        self.train_op2 = optimizer.apply_gradients(train_grads2)

        if sess is None:
            config = tf.ConfigProto(allow_soft_placement=True, )
            sess = tf.Session(config=config)
        self.sess = sess
        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

        if load_path is not None:
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(load_path))

    def train(self,
              trainset: Tuple[list or np.ndarray, list or np.ndarray] or List[list or np.ndarray, list or np.ndarray],
              validationset: Tuple[list or np.ndarray, list or np.ndarray] or List[list or np.ndarray, list or np.ndarray],
              callbacks: None or List[CallBack] = None,
              shuffle: bool = True,
              total_epoch: int = 100,
              second_train_threshold: float = 0.35,
              save_path=None,
              save_interval=10,
              ):

        dataset = Dataset(raw=trainset[0], gnd=trainset[1], batch_size=self.__batchSize__, shuffle=shuffle)
        val_dataset = Dataset(raw=validationset[0], gnd=validationset[1], batch_size=self.__batchSize__, shuffle=False)

        _l2 = 100
        _p2 = np.zeros(shape=(self.__batchSize__, 2), dtype=float)
        best_acc = 0.0
        for current_epoch in range(total_epoch):

            logs_epoch = list()
            pred1_epoch = list()
            pred2_epoch = list()
            anno_epoch = list()

            while not dataset.is_end():
                logs = list()
                mini_batch, total_batch = dataset.report_progress()

                feed_dict = dataset.auto_fetch_with_format({'raw': self.inputHolder, 'gnd': self.annotHolder})

                if callbacks is not None:
                    for callback in callbacks:
                        callback.__on_batch_begin__(current_epoch, mini_batch, feed_dict, logs, tensors=[self.sess, self.logit1, self.logit2, self.cam])

                # optimization
                _, _l1, _p1 = self.sess.run([self.train_op1, self.loss1, tf.nn.softmax(self.logit1, axis=-1)], feed_dict=feed_dict)

                if np.mean(np.equal(np.argmax(_p1, axis=-1), feed_dict[self.annotHolder]).astype(float)) >= second_train_threshold:
                    _, _l2, _p2 = self.sess.run([self.train_op2, self.loss2, tf.nn.softmax(self.logit2, axis=-1)], feed_dict=feed_dict)

                logs.append(f"Epoch [{current_epoch + 1}/{total_epoch}; {current_epoch / float(total_epoch) * 100:.2f}%]")
                logs.append(f"Batch [{mini_batch + 1}/{total_batch + 1}; {mini_batch / float(total_batch + 1) * 100:.2f}%]")
                logs.append(f"Loss: ({_l1:.5f} and {_l2:.5f})")
                logs.append(f"Acc: {np.mean(np.equal(np.argmax(_p1, axis=-1), feed_dict[self.annotHolder]).astype(float)) * 100:.2f}% "
                            f"and {np.mean(np.equal(np.argmax(_p2, axis=-1), feed_dict[self.annotHolder]).astype(float)) * 100:.2f}%")

                anno_epoch.extend(list(feed_dict[self.annotHolder]))
                pred1_epoch.extend(list(np.argmax(_p1, axis=-1)))
                pred2_epoch.extend(list(np.argmax(_p2, axis=-1)))

                if callbacks is not None:
                    for callback in callbacks:
                        callback.__on_batch_end__(current_epoch, mini_batch, feed_dict, logs, tensors=[self.sess, self.logit1, self.logit2, self.cam])

                print(('%s\t / ' * len(logs)) % tuple(logs))

            # For Validation
            val_pred1_epoch = list()
            val_pred2_epoch = list()
            val_anno_epoch = list()
            val_losses1 = list()
            val_losses2 = list()

            while not val_dataset.is_end():
                val_feed_dict = val_dataset.auto_fetch_with_format({'raw': self.inputHolder, 'gnd': self.annotHolder})
                val_loss1, val_p1, val_loss2, val_p2 = self.sess.run(
                    [self.loss1, tf.nn.softmax(self.logit1, axis=-1), self.loss2, tf.nn.softmax(self.logit2, axis=-1)], feed_dict=val_feed_dict)

                val_losses1.append(val_loss1)
                val_losses2.append(val_loss2)

                val_anno_epoch.extend(list(val_feed_dict[self.annotHolder]))
                val_pred1_epoch.extend(list(np.argmax(val_p1, axis=-1)))
                val_pred2_epoch.extend(list(np.argmax(val_p2, axis=-1)))

            val_dataset.reset()

            val_logs_epoch = list()
            val_logs_epoch.append(f"Epoch [{current_epoch + 1}/{total_epoch}; {current_epoch / float(total_epoch) * 100:.2f}%]")
            val_logs_epoch.append(f"Loss: {np.mean(val_losses1):.5f} and {np.mean(val_losses2):.5f}")
            val_logs_epoch.append(f"Acc: {np.mean(np.equal(val_anno_epoch, val_pred1_epoch).astype(float)) * 100:.2f}% "
                                  f"and {np.mean(np.equal(val_anno_epoch, val_pred2_epoch).astype(float)) * 100:.2f}%")

            dataset.reset()

            logs_epoch.append(f"Epoch [{current_epoch + 1}/{total_epoch}; {current_epoch / float(total_epoch):.2f}%]")
            logs_epoch.append(f"Acc: {np.mean(np.equal(anno_epoch, pred1_epoch).astype(float)) * 100:.2f}% "
                              f"and {np.mean(np.equal(anno_epoch, pred2_epoch).astype(float)) * 100:.2f}%")

            if callbacks is not None:
                for callback in callbacks:
                    callback.__on_epoch_end__(current_epoch, dataset, logs_epoch, tensors=[self.sess, self.logit1, self.logit2, self.cam])

            for _ in range(0, 50):
                print("\x1B[H\x1B[J")
            print('{:<15}'.format('Train]'), ('%s\t / ' * len(logs_epoch)) % tuple(logs_epoch))
            print('{:<15}'.format('Validation]'), ('%s\t / ' * len(val_logs_epoch)) % tuple(val_logs_epoch))

            if save_path is not None and current_epoch % save_interval == 0 and np.mean(np.equal(anno_epoch, pred2_epoch).astype(float)) >= best_acc:
                best_acc = np.mean(np.equal(anno_epoch, pred2_epoch).astype(float))
                self.saver.save(self.sess, f"{save_path}/model.ckpt")

    def predict(self, image: np.ndarray):

        assert (len(image.shape) == 3 and image.dtype == np.uint8)

        image = [cv2.resize(image, dsize=(self.__inputShape__[0], self.__inputShape__[1]))]

        feed_dict = {
            self.inputHolder: image
        }

        _labels, _cams = self.sess.run([tf.nn.softmax(self.logit2, axis=-1), self.cam], feed_dict=feed_dict)
        return _labels[0], _cams[0]
