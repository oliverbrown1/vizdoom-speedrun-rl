import tensorflow as tf
tf.profiler.experimental.start('logdir')
print("Profiling started")
tf.profiler.experimental.stop()
print("Profiling stopped")
