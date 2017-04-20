#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 09:43:09 2017

@author: vinnam
"""

import tensorflow as tf

init = tf.placeholder(dtype = 'float32', shape = [None])
var = tf.Variable(init, validate_shape = False, name ='a')
obj = tf.reduce_sum(tf.square(var))
obj1 = tf.reduce_sum(tf.square(init))
grad = tf.gradients(obj, var)
grad1 = tf.gradients(obj1, init)
opt = tf.train.AdamOptimizer()
opt.minimize(obj, var_list = [var])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer(), {init : [1., 2., 3., 5.]})
sess.run(grad, {init : [1., 2., 3., 5.]})
sess.run(grad1, {init : [1., 2., 4.]})

sess.run(opt, )

tf.contrib.opt.ExternalOptimizerInterface
tf.contrib.opt.ScipyOptimizerInterface(obj, var_list = [var])

def _tf_create_slot_var(primary, val, scope):
  """Helper function for creating a slot variable."""

  from tensorflow.python.ops import variables
  slot = variables.Variable(val, name=scope, trainable=False, validate_shape=primary.get_shape().is_fully_defined())
  # pylint: disable=protected-access
  if isinstance(primary, variables.Variable) and primary._save_slice_info:
    # Primary is a partitioned variable, so we need to also indicate that
    # the slot is a partitioned variable.  Slots have the same partitioning
    # as their primaries.
    real_slot_name = scope[len(primary.op.name + "/"):-1]
    slice_info = primary._save_slice_info
    slot._set_save_slice_info(variables.Variable.SaveSliceInfo(
        slice_info.full_name + "/" + real_slot_name,
        slice_info.full_shape[:],
        slice_info.var_offset[:],
        slice_info.var_shape[:]))
  # pylint: enable=protected-access
  return slot


def _tf_create_zeros_slot(primary, name, dtype=None, colocate_with_primary=True):
  """Create a slot initialized to 0 with same shape as the primary object.

  Args:
    primary: The primary `Variable` or `Tensor`.
    name: Name to use for the slot variable.
    dtype: Type of the slot variable.  Defaults to the type of `primary`.
    colocate_with_primary: Boolean.  If True the slot is located
      on the same device as `primary`.

  Returns:
    A `Variable` object.
  """
  if dtype is None:
    dtype = primary.dtype
  from tensorflow.python.ops import array_ops
  val = array_ops.zeros(
      primary.get_shape().as_list() if primary.get_shape().is_fully_defined() else tf.shape(primary),
      dtype=dtype)
  from tensorflow.python.training import slot_creator
  return slot_creator.create_slot(primary, val, name, colocate_with_primary=colocate_with_primary)


def monkey_patch_tf_slot_creator():
    """
    The TensorFlow optimizers cannot handle variables with unknown shape.
    We hack this.
    """
    from tensorflow.python.training import slot_creator
    slot_creator._create_slot_var = _tf_create_slot_var
    slot_creator.create_zeros_slot = _tf_create_zeros_slot
    
monkey_patch_tf_slot_creator()