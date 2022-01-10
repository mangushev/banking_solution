
#TODO:

import tensorflow as tf
import tensorflow_addons as tfa

import math
import six

import numpy as np

class Dense(tf.Module):
  def __init__(self, input_width, output_size, activation=None, stddev=1.0, name=''):
    super(Dense, self).__init__()
    self.w = tf.Variable(
      tf.random.truncated_normal([input_width, output_size], stddev=stddev), name=name + '_w')
    self.b = tf.Variable(tf.zeros([output_size]), name=name+'_b')
    self.activation = activation
    self.input_width = input_width
    self.output_size = output_size
  def __call__(self, x):
    input_shape = x.shape

    if len(input_shape) != 2 and len(input_shape) != 3:
      raise ValueError("input shape rank {} shuld be 2 or 3".format(len(input_shape)))

    if len(input_shape) == 3:
      if self.input_width != input_shape[2]:
        raise ValueError("widths do not match {} {}".format(self.input_width, input_shape[2]))
      x = tf.reshape(x, [-1, self.input_width])
    else:
      if self.input_width != input_shape[1]:
        raise ValueError("widths do not match {} {}".format(self.input_width, input_shape[1]))

    y = tf.matmul(x, self.w) + self.b
    if (self.activation is not None):
      y = self.activation(y)

    if len(input_shape) == 3:
      return tf.reshape(y, [-1, input_shape[1], self.output_size])

    return y

def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, dropout_prob)
  return output

def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  #if name is None:
  #  name = tensor.name

  if expected_rank is not None:
    #assert_rank(tensor, expected_rank, name)
    if not len(tensor.shape) in expected_rank:
      raise ValueError("input shape rank {} should be {}".format(len(tensor.shape), expected_rank))

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape

def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=[2])
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask

class AttentionLayer(tf.Module):
  def __init__(self, attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    from_width=None,
                    to_width=None):
    super(AttentionLayer, self).__init__()

#  """Performs multi-headed attention from `from_tensor` to `to_tensor`.
#
#  This is an implementation of multi-headed attention based on "Attention
#  is all you Need". If `from_tensor` and `to_tensor` are the same, then
#  this is self-attention. Each timestep in `from_tensor` attends to the
#  corresponding sequence in `to_tensor`, and returns a fixed-with vector.
#
#  This function first projects `from_tensor` into a "query" tensor and
#  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
#  of tensors of length `num_attention_heads`, where each tensor is of shape
#  [batch_size, seq_length, size_per_head].
#
#  Then, the query and key tensors are dot-producted and scaled. These are
#  softmaxed to obtain attention probabilities. The value tensors are then
#  interpolated by these probabilities, then concatenated back to a single
#  tensor and returned.
#
#  In practice, the multi-headed attention are done with transposes and
#  reshapes rather than actual separate tensors.
#
#  Args:
#    from_tensor: float Tensor of shape [batch_size, from_seq_length,
#      from_width].
#    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
#    attention_mask: (optional) int32 Tensor of shape [batch_size,
#      from_seq_length, to_seq_length]. The values should be 1 or 0. The
#      attention scores will effectively be set to -infinity for any positions in
#      the mask that are 0, and will be unchanged for positions that are 1.
#    num_attention_heads: int. Number of attention heads.
#    size_per_head: int. Size of each attention head.
#    query_act: (optional) Activation function for the query transform.
#    key_act: (optional) Activation function for the key transform.
#    value_act: (optional) Activation function for the value transform.
#    attention_probs_dropout_prob: (optional) float. Dropout probability of the
#      attention probabilities.
#    initializer_range: float. Range of the weight initializer.
#    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
#      * from_seq_length, num_attention_heads * size_per_head]. If False, the
#      output will be of shape [batch_size, from_seq_length, num_attention_heads
#      * size_per_head].
#    batch_size: (Optional) int. If the input is 2D, this might be the batch size
#      of the 3D version of the `from_tensor` and `to_tensor`.
#    from_seq_length: (Optional) If the input is 2D, this might be the seq length
#      of the 3D version of the `from_tensor`.
#    to_seq_length: (Optional) If the input is 2D, this might be the seq length
#      of the 3D version of the `to_tensor`.
#
#  Returns:
#    float Tensor of shape [batch_size, from_seq_length,
#      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
#      true, this will be of shape [batch_size * from_seq_length,
#      num_attention_heads * size_per_head]).
#
#  Raises:
#    ValueError: Any of the arguments or tensor shapes are invalid.
#  """

    self.attention_mask = attention_mask
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.do_return_2d_tensor = do_return_2d_tensor
    self.batch_size = batch_size
    self.from_seq_length=from_seq_length
    self.to_seq_length=to_seq_length
    
    # `query_layer` = [B*F, N*H]
    self.query_layer = Dense(
      from_width,
      num_attention_heads * size_per_head,
      activation=query_act,
      name="query",
      stddev=initializer_range)

    # `key_layer` = [B*T, N*H]
    self.key_layer = Dense(
      to_width,
      num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      stddev=initializer_range)

    # `value_layer` = [B*T, N*H]
    self.value_layer = Dense(
      to_width,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      stddev=initializer_range)

  def __call__(self, from_tensor, to_tensor):

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
      output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

      output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
      return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
      raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
      batch_size = from_shape[0]
      from_seq_length = from_shape[1]
      to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
      if (self.batch_size is None or self.from_seq_length is None or self.to_seq_length is None):
        raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query = self.query_layer(from_tensor_2d)

    # `key_layer` = [B*T, N*H]
    key = self.key_layer(to_tensor_2d)

    # `value_layer` = [B*T, N*H]
    value = self.value_layer(to_tensor_2d)

    # `query_layer` = [B, N, F, H]
    query = transpose_for_scores(query, self.batch_size,
                                     self.num_attention_heads, self.from_seq_length,
                                     self.size_per_head)

    # `key_layer` = [B, N, T, H]
    key = transpose_for_scores(key, self.batch_size, self.num_attention_heads,
                                   self.to_seq_length, self.size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query, key, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(self.size_per_head)))
    if self.attention_mask is not None:
      # `attention_mask` = [B, 1, F, T]
      attention_mask = tf.expand_dims(self.attention_mask, axis=[1])

      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      #adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -1e12
      adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      attention_scores += adder
      #print (attention_scores)

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, self.attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value = tf.reshape(
      value,
      [self.batch_size, self.to_seq_length, self.num_attention_heads, self.size_per_head])

    # `value_layer` = [B, N, T, H]
    value = tf.transpose(value, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context = tf.matmul(attention_probs, value)

    # `context_layer` = [B, F, N, H]
    context = tf.transpose(context, [0, 2, 1, 3])

    if self.do_return_2d_tensor:
      # `context_layer` = [B*F, N*H]
      context = tf.reshape(
        context,
        [self.batch_size * self.from_seq_length, self.num_attention_heads * self.size_per_head])
    else:
      # `context_layer` = [B, F, N*H]
      context = tf.reshape(
        context,
        [self.batch_size, self.from_seq_length, self.num_attention_heads * self.size_per_head])

    return context

class TransformerLayer(tf.Module):
  def __init__(self, batch_size=None,
                     from_seq_length=None,
                     to_seq_length=None,
                     attention_mask=None,
                     hidden_size=768,
                     num_hidden_layers=12,
                     num_attention_heads=12,
                     intermediate_size=768,
                     intermediate_act_fn=tf.nn.relu,
                     hidden_dropout_prob=0.1,
                     attention_probs_dropout_prob=0.1,
                     initializer_range=0.02):
    super(TransformerLayer, self).__init__()
 
    self.hidden_dropout_prob = hidden_dropout_prob

    self.attention_head = AttentionLayer(
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=int(hidden_size / num_attention_heads),
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=from_seq_length,
              to_seq_length=to_seq_length,
              from_width=hidden_size,
              to_width=hidden_size)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
    self.projection_layer = Dense(
            hidden_size,
            hidden_size,
            stddev=initializer_range,
            name='projection_layer')

    # The activation is only applied to the "intermediate" hidden layer.
    self.intermediate_layer = Dense(
            hidden_size,
            intermediate_size,
            activation=intermediate_act_fn,
            stddev=initializer_range,
            name='intermediate_layer')

    # Down-project back to `hidden_size` then add the residual.
    self.down_projection_layer = Dense(
            intermediate_size,
            hidden_size,
            stddev=initializer_range,
            name='down_projection_layer')

  def __call__(self, layer_input):
    attention_heads = []
    attention_heads.append(self.attention_head(layer_input, layer_input))

    attention_output = None
    if len(attention_heads) == 1:
      attention_output = attention_heads[0]
    else:
      # In the case where we have other sequences, we just concatenate
      # them to the self-attention head before the projection.
      attention_output = tf.concat(attention_heads, axis=-1)

    # Run a linear projection of `hidden_size` then add a residual
    # with `layer_input`.
    attention_output = self.projection_layer(attention_output)
    attention_output = dropout(attention_output, self.hidden_dropout_prob)
    attention_output = tfa.layers.GroupNormalization(groups = 1)(attention_output + layer_input)

    # The activation is only applied to the "intermediate" hidden layer.
    intermediate_output = self.intermediate_layer(attention_output)

    # Down-project back to `hidden_size` then add the residual.
    layer_output = self.down_projection_layer(intermediate_output)
    layer_output = dropout(layer_output, self.hidden_dropout_prob)
    return tfa.layers.GroupNormalization(groups = 1)(layer_output + attention_output)

class Transformer(tf.Module):
  def __init__(self, batch_size=None,
                      from_seq_length=None,
                      to_seq_length=None,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=768,
                      intermediate_act_fn=tf.nn.relu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
    super(Transformer, self).__init__()

#  """Multi-headed, multi-layer Transformer from "Attention is All You Need".
#
#  This is almost an exact implementation of the original Transformer encoder.
#
#  See the original paper:
#  https://arxiv.org/abs/1706.03762
#
#  Also see:
#  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
#
#  Args:
#    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
#    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
#      seq_length], with 1 for positions that can be attended to and 0 in
#      positions that should not be.
#    hidden_size: int. Hidden size of the Transformer.
#    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
#    num_attention_heads: int. Number of attention heads in the Transformer.
#    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
#      forward) layer.
#    intermediate_act_fn: function. The non-linear activation function to apply
#      to the output of the intermediate/feed-forward layer.
#    hidden_dropout_prob: float. Dropout probability for the hidden layers.
#    attention_probs_dropout_prob: float. Dropout probability of the attention
#      probabilities.
#    initializer_range: float. Range of the initializer (stddev of truncated
#      normal).
#    do_return_all_layers: Whether to also return all layers or just the final
#      layer.
#
#  Returns:
#    float Tensor of shape [batch_size, seq_length, hidden_size], the final
#    hidden layer of the Transformer.
#
#  Raises:
#    ValueError: A Tensor shape or parameter is invalid.
#  """
    if hidden_size % num_attention_heads != 0:
      raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  #  attention_head_size = int(hidden_size / num_attention_heads)
  #  input_shape = get_shape_list(input_tensor, expected_rank=[3])
  #  batch_size = input_shape[0]
  #  seq_length = input_shape[1]
  #  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  #  if input_width != hidden_size:
  #    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
  #                   (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  
    self.do_return_all_layers = do_return_all_layers

    self.transformer_layers = []

    for _ in range(num_hidden_layers):
      self.transformer_layers.append(TransformerLayer(
                      batch_size=batch_size,
                      from_seq_length=from_seq_length,
                      to_seq_length=to_seq_length,
                      attention_mask=attention_mask,
                      hidden_size=hidden_size,
                      num_hidden_layers=num_hidden_layers,
                      num_attention_heads=num_attention_heads,
                      intermediate_size=intermediate_size,
                      intermediate_act_fn=intermediate_act_fn,
                      hidden_dropout_prob=hidden_dropout_prob,
                      attention_probs_dropout_prob=attention_probs_dropout_prob,
                      initializer_range=initializer_range))

  def __call__(self, input_tensor):
    input_shape = get_shape_list(input_tensor, expected_rank=[3])

    all_layer_outputs = []
    prev_output = reshape_to_matrix(input_tensor)
    for transformer_layer in self.transformer_layers:
      prev_output = transformer_layer(prev_output)
      all_layer_outputs.append(prev_output)

    if self.do_return_all_layers:
      final_outputs = []
      for layer_output in all_layer_outputs:
        final_output = reshape_from_matrix(layer_output, input_shape)
        final_outputs.append(final_output)
      return final_outputs
    else:
      final_output = reshape_from_matrix(prev_output, input_shape)
      return final_output

class ExpenseEstimator(tf.Module):
  #   B = batch size (number of sequences)
  #   n = lookback history
  #   o = output sequence size (1 day/week, 7 days/weeks)
  #   d - hidden size
  #   f = number of features
  #   e = number of estimated features
  def __init__(self, batch_size,
                     lookback_history,
                     input_width,
                     hidden_size=64,
                     num_hidden_layers=2,
                     num_attention_heads=2,
                     activation_fn=tf.nn.sigmoid,
                     dropout_prob=0.1,
                     initializer_range=1.0,
                     is_training=False):
    super(ExpenseEstimator, self).__init__()

    if is_training == False:
      dropout_prob = 0.0   
 
    self.alignment_layer = Dense(input_width, 
    			hidden_size,
            		stddev=initializer_range,
    			name='estimator_alignment_layer')

    self.encoder_position_table = tf.Variable(
      tf.random.truncated_normal([lookback_history, hidden_size], stddev=initializer_range), name='encoder_position_table')

    attention_mask = tf.tile(tf.expand_dims(tf.linalg.band_part(tf.ones([lookback_history, lookback_history], dtype=tf.float32), -1, 0), 0), [batch_size, 1, 1])

    self.transformer_layer = Transformer(
                        batch_size=batch_size,
                        from_seq_length=lookback_history,
                        to_seq_length=lookback_history,
                        attention_mask=attention_mask,
                        hidden_size=hidden_size,
                        num_hidden_layers=num_hidden_layers,
                        num_attention_heads=num_attention_heads,
                        intermediate_size=hidden_size,
                        hidden_dropout_prob=dropout_prob,
                        attention_probs_dropout_prob=dropout_prob,
                        #initializer_range=0.2,
                        initializer_range=0.02,
                        do_return_all_layers=False)
  
    self.output_layer = Dense(hidden_size, 
			1, 
			activation=activation_fn,
            		stddev=initializer_range,
			name='estimator_output_layer')

  def __call__(self, x):
    #(B, n, f) --> (B, n, d)
    output = self.alignment_layer(x)
    #(B, n, d) + (B, n, d) --> (B, n, d) --> (B, n, d)
    return self.transformer_layer(output + tf.expand_dims(self.encoder_position_table, 0))
