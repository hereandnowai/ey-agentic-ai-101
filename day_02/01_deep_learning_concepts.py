import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging (1)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN to avoid AVX/FMA warnings

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings

import absl.logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)  # Suppress absl logging

import numpy as np
import tensorflow as tf

tf

