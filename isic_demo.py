#Copyright 2015 The TensorFlow Authors. All Rights Reserved.  #
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cv2 as cv
from cifar import cifar10
from isic_cnn import utils as u
import os.path as op

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
                           """Directory where to read RGB model checkpoints.""")
tf.app.flags.DEFINE_string('RGB_dir', '/home/charlie/cifar_isic_checkpoints/RGB_train',
                           """Directory where to read RGB model checkpoints.""")
tf.app.flags.DEFINE_string('FFT_dir', '/home/charlie/cifar_isic_checkpoints/FFT_train',
                           """Directory where to read FFT model checkpoints.""")
tf.app.flags.DEFINE_string('HSV_dir', '/home/charlie/cifar_isic_checkpoints/HSV_train',
                           """Directory where to read HSV model checkpoints.""")
tf.app.flags.DEFINE_string('images_dir', '/home/charlie/projects/demo_isic_cnn/images', """Directory that the demo images reside""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, model_ckpt):
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(model_ckpt)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
      guess = sess.run([logits])
      print('%s' % (datetime.now()))
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    return guess


def evaluate(image,summary_dir):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:


    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = tf.nn.softmax(cifar10.inference(image))

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

    while True:
      return eval_once(saver, summary_writer, summary_dir)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    print(FLAGS.images_dir)
    rgbim = cv.imread(op.join(FLAGS.images_dir,"image1.png"))
    fftim = u.imageSet.to_FFT(rgbim)
    hsvim = u.imageSet.to_HSV(rgbim)

    tf_rgbim = tf.cast(rgbim, tf.float32)
    tf_fftim = tf.cast(fftim, tf.float32)
    tf_hsvim = tf.cast(hsvim, tf.float32)

    inf_rgb = evaluate(rgbim,FLAGS.RGB_dir)
    inf_fft = evaluate(fftim,FLAGS.FFT_dir)
    inf_hsv = evaluate(hsvim,FLAGS.HSV_dir)

    print('[RGB, FFT, HSV]:[%s, %s, %s]' % (inf_rgb, inf_fft, inf_hsv))

if __name__ == '__main__':
    tf.app.run()
