## linear regression example
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

oecd_bli = pd.read_csv("oecd_bli.2015.csv", thousands = ",")