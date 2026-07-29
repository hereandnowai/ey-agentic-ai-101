import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.decomposition import PCA