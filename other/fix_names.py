import os
import pathlib
import pickle

filenames = os.listdir('/home/tamas/repos/psycholinguistic_API_UC/twitter/collected_tweets_python_twitter')

for f in filenames:
    path = "/home/tamas/repos/psycholinguistic_API_UC/twitter/collected_tweets_python_twitter/" + f
    with open(path, "rb") as d:
        dataset = pickle.load(d)

    new_name = f.lower().replace(".pickle", "").replace(" ", "_").replace(":", "_").replace(".", "_") + ".pickle"
    new_path = "/home/tamas/repos/psycholinguistic_API_UC/twitter/collected_tweets_python_twitter/" + new_name
    with open(new_path, "wb") as d:
        pickle.dump(dataset, d)

