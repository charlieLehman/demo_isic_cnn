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
tf.app.flags.DEFINE_string('images_dir', '/home/charlie/Dropbox/demo/',
                           """Directory that the demo images reside""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")
global guess
global benign
guess = []
benign = []



def eval_once(saver, summary_writer, model_ckpt, inference, logits, top_k_op):
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
      global guess
      global benign

      guess.append(sess.run([logits])[0][0])
      benign.append(sess.run([top_k_op])[0][0])

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
  return guess



def evaluate(image,summary_dir):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:

    height = 220
    width = 220

    # Build a Graph that computes the logits predictions from the
    # inference model.
    
    tf_im = tf.image.per_image_whitening(tf.image.resize_image_with_crop_or_pad(tf.cast(image, tf.float32),width,height))
    bimage, label_batch = tf.train.batch(
        [tf_im, 0],
        batch_size=1,
        num_threads=1,
        capacity=4 )

    inference = cifar10.inference(bimage)
    logits = tf.nn.softmax(inference)
    top_k_op = tf.nn.in_top_k(logits,[0],1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, summary_dir,inference, logits, top_k_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)

def conditional_switch(logic_vector):
    switch = {
         '0b111':"99.91",
         '0b110':"83.33",
         '0b101':"84",
         '0b100':"10",
         '0b011':"84.91",
         '0b001':"10.64",
         '0b010':"6.67",
         '0b000':"0"
    }
    return switch.get(logic_vector,"none")

def main(argv=None):  # pylint: disable=unused-argument
    rgbim = cv.imread(op.join(FLAGS.images_dir,"image.jpg"))
    fftim = u.imageSet.to_FFT(rgbim)
    hsvim = u.imageSet.to_HSV(rgbim)


    evaluate(rgbim,FLAGS.RGB_dir)
    evaluate(fftim,FLAGS.FFT_dir)
    evaluate(hsvim,FLAGS.HSV_dir)

    
    mean = (guess[0]+guess[1]+guess[2])/3
    benign_case = '0b' + ''.join(['1' if x else '0' for x in benign])
    print('There is a %s %% confidence this is Benign' % (conditional_switch(benign_case)))
    
    print('Is it benign? [RGB, FFT, HSV]')
    print(benign)

    print('                 [ BENI | MALI ]' )
    print('RGB distribution:[ %0.2f | %0.2f ]' % (guess[0][0], guess[0][1]))
    print('FFT distribution:[ %0.2f | %0.2f ]' % (guess[1][0], guess[1][1]))
    print('HSV distribution:[ %0.2f | %0.2f ]' % (guess[2][0], guess[2][1]))


if __name__ == '__main__':
    tf.app.run()
