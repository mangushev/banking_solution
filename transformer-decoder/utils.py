
import tensorflow as tf

class Scaler(tf.Module):
  def __init__(self, X_min, X_max, range_min, range_max):
    super(Scaler, self).__init__()
    self.X_min = X_min
    self.X_max = X_max
    self.range_min = range_min
    self.range_max = range_max
  def transform(self, X):
    return X / self.X_max
    X_std = (X - self.X_min) / (self.X_max - self.X_min)
    return X_std * (self.range_max - self.range_min) + self.range_min

  def inverse_transform(self, X_scaled):
    return X_scaled * self.X_max
    X_std = (X_scaled - self.range_min) / (self.range_max - self.range_min)
    return X_std / (self.X_max - self.X_min) + self.X_min
