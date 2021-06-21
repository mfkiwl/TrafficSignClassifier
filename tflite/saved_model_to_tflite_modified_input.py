import tensorflow as tf 

savepath = "models/my_resnet640/saved_model"

model = tf.saved_model.load(savepath)
concrete_func = model.signatures[
  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([None, 640, 640, 3])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

tflite_model = converter.convert()
open("models/exported_resnet640.tflite", "wb").write(tflite_model)
