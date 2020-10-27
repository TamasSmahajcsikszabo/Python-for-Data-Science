import streamlit as st
import numpy as np
import pandas as pd
import psycholinguistics as psyling
import TwitterSearch as TS
import pandas as pd
from datetime import date, datetime
import pickle
import pathlib
import json
import os


def get_data(limit: int = 100):
    """Gets all stored data and returns them as JSON
    The function filters out duplicates.
    :limit: integer, how many records should be retrieved
    :returns: JSON string

    """
    data = []
    path = str(pathlib.Path(__file__).parent.absolute()) + "/../../psycholinguistic_API_UC/twitter/" +'collected_tweets/'
    path_list = os.listdir(path)
    for obj in path_list:
                pathname = path + obj
                with open(pathname, 'rb') as f:
                    dataset = pickle.load(f)
                data.append(dataset)
    data = pd.concat(data)
    print('Total records found: ' + str(len(data)))

    data = data.drop_duplicates(subset=['id'])
    data = data.tail(limit)
    data =  data.to_dict(orient="records")

    return data



st.title("Twitter demo")
data = get_data(10000)
st.dataframe(data, height = 800)

