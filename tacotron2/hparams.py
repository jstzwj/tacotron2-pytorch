import tensorflow as tf

import six

class HParams(object):
  _HAS_DYNAMIC_ATTRIBUTES = True  # Required for pytype checks.

  def __init__(self, model_structure=None, **kwargs):
    self._hparam_types = {}
    self._model_structure = model_structure
    for name, value in six.iteritems(kwargs):
      self.add_hparam(name, value)

  def add_hparam(self, name, value):
    if getattr(self, name, None) is not None:
      raise ValueError('Hyperparameter name is reserved: %s' % name)
    if isinstance(value, (list, tuple)):
      if not value:
        raise ValueError(
            'Multi-valued hyperparameters cannot be empty: %s' % name)
      self._hparam_types[name] = (type(value[0]), True)
    else:
      self._hparam_types[name] = (type(value), False)
    setattr(self, name, value)

  def set_hparam(self, name, value):
    param_type, is_list = self._hparam_types[name]
    if isinstance(value, list):
      if not is_list:
        raise ValueError(
            'Must not pass a list for single-valued parameter: %s' % name)
      setattr(self, name, [
          _cast_to_type_if_compatible(name, param_type, v) for v in value])
    else:
      if is_list:
        raise ValueError(
            'Must pass a list for multi-valued parameter: %s.' % name)
      setattr(self, name, _cast_to_type_if_compatible(name, param_type, value))

  def del_hparam(self, name):
    if hasattr(self, name):
      delattr(self, name)
      del self._hparam_types[name]

  def parse(self, values):
    type_map = {}
    for name, t in self._hparam_types.items():
      param_type, _ = t
      type_map[name] = param_type

    values_map = parse_values(values, type_map)
    return self.override_from_dict(values_map)

  def override_from_dict(self, values_dict):
    for name, value in values_dict.items():
      self.set_hparam(name, value)
    return self

  def set_model_structure(self, model_structure):
    self._model_structure = model_structure

  def get_model_structure(self):
    return self._model_structure

  def to_json(self, indent=None, separators=None, sort_keys=False):
    def remove_callables(x):
      """Omit callable elements from input with arbitrary nesting."""
      if isinstance(x, dict):
        return {k: remove_callables(v) for k, v in six.iteritems(x)
                if not callable(v)}
      elif isinstance(x, list):
        return [remove_callables(i) for i in x if not callable(i)]
      return x
    return json.dumps(
        remove_callables(self.values()),
        indent=indent,
        separators=separators,
        sort_keys=sort_keys)

  def parse_json(self, values_json):
    values_map = json.loads(values_json)
    return self.override_from_dict(values_map)

  def values(self):
    return {n: getattr(self, n) for n in self._hparam_types.keys()}

  def get(self, key, default=None):
    """Returns the value of `key` if it exists, else `default`."""
    if key in self._hparam_types:
      # Ensure that default is compatible with the parameter type.
      if default is not None:
        param_type, is_param_list = self._hparam_types[key]
        type_str = 'list<%s>' % param_type if is_param_list else str(param_type)
        fail_msg = ("Hparam '%s' of type '%s' is incompatible with "
                    'default=%s' % (key, type_str, default))

        is_default_list = isinstance(default, list)
        if is_param_list != is_default_list:
          raise ValueError(fail_msg)

        try:
          if is_default_list:
            for value in default:
              _cast_to_type_if_compatible(key, param_type, value)
          else:
            _cast_to_type_if_compatible(key, param_type, default)
        except ValueError as e:
          raise ValueError('%s. %s' % (fail_msg, e))

      return getattr(self, key)

    return default

  def __contains__(self, key):
    return key in self._hparam_types

  def __str__(self):
    return str(sorted(self.values().items()))

  def __repr__(self):
    return '%s(%s)' % (type(self).__name__, self.__str__())

  @staticmethod
  def _get_kind_name(param_type, is_list):
    if issubclass(param_type, bool):
      # This check must happen before issubclass(param_type, six.integer_types),
      # since Python considers bool to be a subclass of int.
      typename = 'bool'
    elif issubclass(param_type, six.integer_types):
      # Setting 'int' and 'long' types to be 'int64' to ensure the type is
      # compatible with both Python2 and Python3.
      typename = 'int64'
    elif issubclass(param_type, (six.string_types, six.binary_type)):
      # Setting 'string' and 'bytes' types to be 'bytes' to ensure the type is
      # compatible with both Python2 and Python3.
      typename = 'bytes'
    elif issubclass(param_type, float):
      typename = 'float'
    else:
      raise ValueError('Unsupported parameter type: %s' % str(param_type))

    suffix = 'list' if is_list else 'value'
    return '_'.join([typename, suffix])

def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        training_files='filelists/ljs_audio_text_train_filelist.txt',
        validation_files='filelists/ljs_audio_text_val_filelist.txt',
        text_cleaners=['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=25565,
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=4,
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
