import os
import sys
import hashlib
import pickle
import re
import typing
import spacy
import pandas as pd
import numpy as np
from spellchecker import SpellChecker
from psycholinguistics.core import get_sentiment_score
from datetime import datetime
from util.cloud_connection import upload_object, download_object, bucket_name, list_files, create_container
from psycholinguistics.psychological import get_words, get_word_count, get_ratios, estimate_attribute
nlp = spacy.load('en_core_web_sm')

# from psycholinguistics.linguistics import get_readability_scores


def get_security_token() -> str:
    """
    A function to get a security token
    """

    h = hashlib.new('ripemd160')
    h.update(b"The spice must flow")
    return str(h.hexdigest())


def check_security_token(request: typing.Any) -> None:
    """
    Checks if the request has the necessary security token as a passed along
    header parameter by the name 'token'.
    """
    inbound_token = request.headers['Token']
    match = inbound_token == get_security_token()
    if match:
        pass
    else:
        sys.exit("Invalid security token!")


def load_local_vocabulary(vocab: str, client: str = "s3") -> typing.List[str]:
    """
    Loads locally stored [Pickle] vocabulary lists
    :vocab: the string name of the vocabulary
    :returns: the list of words from the actual vocabularies
    """
    filename = 'vocabularies/'+vocab+'.vocab'
    try:
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                vocabulary = pickle.load(f)
            print('{0} vocabulary found locally and loaded'.format(vocab))
        else:
            download_object(client=client, file_name=filename)
            with open(filename, 'rb') as f:
                vocabulary = pickle.load(f)
            print('{0} vocabulary downloaded successfully'.format(vocab))
    except BaseException as ex:
        print('Something went wrong: \n')
        print(ex)
    return vocabulary


def get_unintelligible_words(text: str, absolute: bool = False) -> float:
    spelling = SpellChecker()
    unintelligble = len(spelling.unknown(get_words(text)))
    total = len(get_words(text))

    if absolute:
        return float(unintelligble)
    else:
        return unintelligble / total


def estimate_features(
        text: str, per_word: bool = True, absolute: bool = False,
        custom_vocabularies: typing.List[str] = []) -> pd.DataFrame:
    """
    Estimates the psycholinguistic and custom features used in the given request

    :text: the text input
    :per_word: Boolean, should the sentiment estimate be per word average, or cumulative
    :absolute: Boolean, should the estimation be ratio or absolute count of words
    :custom_vocabularies: string list of names of additional custom vocabularies to be used
    :returns: a Pandas DataFrame of all the estimated results
    """
    psyling_estimates = pd.DataFrame(
        estimate_attribute(
            "full_profile",
            text,
            print_out=True,
            absolute=absolute,
            only_components=True),
        index=[0])

    sentiment_scores_per_word = list(
        get_sentiment_score(
            text, per_word=per_word, rescaled=True).values())[0]
    word_count_estimate = get_word_count(text)
    unintelligible = get_unintelligible_words(text, absolute=True)

    psyling_estimates['sentiment_scores_per_word'] = sentiment_scores_per_word
    psyling_estimates['word_count_estimate'] = word_count_estimate
    psyling_estimates['unintelligible'] = unintelligible
    if custom_vocabularies != ['']:
        for vocab in custom_vocabularies:
            psyling_estimates[vocab] = get_ratios(
                text, item_list=load_local_vocabulary(vocab),
                absolute=absolute, pattern=True, j=2, no_strip=True)

    return psyling_estimates


def reorganize_columns(input_data: pd.DataFrame) -> pd.DataFrame:
    """Makes columns of new training data aligned with existing training set

    :input_data: input Pandas dataframe
    :returns: pandas DataFrame of new data

    """
    file_name = 'data/ml_data'
    with open(file_name, 'rb') as f:
        ml_data_columns = pickle.load(f).columns.tolist()
    output_data = input_data.reindex(columns=ml_data_columns)
    return output_data


def impute_missing_feature(
        estimates: pd.DataFrame, update: bool = False, client: str = 's3') -> pd.DataFrame:
    """
    Imputes custom vocabulary estimates that weren't given in the input estimate,
    but that are part of the actual trained model.
    Imputed fields are set to 0.
    Good for show/investigate predictor [especially custom vocabulary] impact
    on predictions.

    :estimates: a Pandas DataFrame of input predictors
    :update: Boolean, if True, the function updates the existing training data from the cloud
    :client: string name of cloud client, only relevant, if update is set to true
    :returns: a Pandas DataFrame with additional imputed columns
    """

    existing_names = estimates.columns.tolist()

    file_name = 'data/ml_data'
    if update == True:
        download_object(client=client, file_name=file_name)

    with open(file_name, 'rb') as f:
        ml_data_columns = pickle.load(f).columns.tolist()

    missing_names = [m not in existing_names for m in ml_data_columns]
    missing_names_index = [int(m)
                           for m in pd.Series(range(len(missing_names)))
                           [missing_names]]
    for i in missing_names_index:
        missing_col = ml_data_columns[i]
        estimates[missing_col] = 0.0

    estimates = reorganize_columns(estimates)
    return estimates


def strip_quote_vec(
        vector: typing.List[str],
        categories: typing.List[str] = ["urgent", "not_urgent"],
        replace_value: str = "") -> typing.List[str]:
    """
    Strips quotation marks from a list of strings

    :vector: input string list
    :categories: if the vector serves labeling purpose [i.e. response variable]
    what should be the categories
    :replace_value: what should be the replacement
    :returns: the modified list of strings
    """
    mask = [token not in categories for token in vector]
    to_replace_values = vector[mask]

    modified_values = [value.replace('"', "") for value in to_replace_values]
    vector[mask] = modified_values
    return vector


def strip_quote(text: str) -> str:
    """Strips quotation marks from string.
    :returns: the striped string
    """
    return text.replace('"', "")


def parse_formula(formula: str) -> typing.List[str]:
    """
    Splits formula string to components
    :formula: string of formula
    :returns: a list of strings
    """
    return formula.split(" ")


def count_components(formula: str, value_range: typing.List[int] = [
                     0, 1]) -> typing.List[int]:
    """
    //depricated// -used only in early stages of the API
    Counts components of the list: how many positive and negative components a
    formula has

    :formula: string formula
    :value_range: what is the actual value range of the component in the data
    :returns: possible, theoretical minimum and maximum range of a formula
    """
    components = parse_formula(formula)
    negatives = 0
    positives = 0
    for i in range(len(components)):
        if components[i-1] == "minus":
            negatives = negatives + 1

    operators = ['plus', 'minus']
    components_filered = [c for c in components if c not in operators]
    positives = len(components_filered) - negatives
    if negatives == 0:
        min_value = max(value_range) * negatives * (-1)
    else:
        min_value = max(value_range) * negatives * (-1) / negatives

    if positives == 0:
        max_value = max(value_range) * positives
    else:
        max_value = max(value_range) * positives / positives

    return([min_value, max_value])


def rescale(x: float, rescale_min: int = 0, rescale_max: int = 1,
            formula: str = "", downgrade_factor: float = 0.5) -> float:
    """
    Rescale to a given minimum and maximum range
    :x: input float
    :rescale_min: what should be the rescale min
    :rescale_max: what should be the rescale max
    :formula: when given, min-max range is taken from the min-max estimate
    :downgrade_factor: custom, manual factor for weighting
    :returns: rescaled score
    """
    range_x = count_components(formula)
    min_x = min(range_x)
    max_x = max(range_x)
    x_rescaled = rescale_min + ((x - min_x)
                                * (rescale_max - rescale_min)) / (max_x - min_x)
    return x_rescaled


def sigmoid_f(x: float) -> float:
    """
    Performs sigmoid transformation on input score
    :x: input score
    :returns: transformed score
    """
    return 1 / (1 + np.exp(-x))


def yield_categories(n_categories: int = 4, custom_labels: str = None,
                     thresholds : typing.List[float] = None) -> dict:
    """
    Helper function for estimate_formula() to yield category names and score thresholds as dict
    :n_categories: integer,indicates the number of custom categories
    :custom_labels: list of string names of custom category names
    :returns: Dict of category names and thresholds
    """
    categories = dict()
    if thresholds is None:
        step = 1 / n_categories
        values = [0]

        for i in range(n_categories-1):
            values.append(values[i] + step)
        values.append(1)
    else:
        values = [0]
        for i in range(len(thresholds)):
            values.append(thresholds[i])
        values.append(1)
        n_categories = len(thresholds) + 1

    if custom_labels is not None:
        if len(custom_labels) == n_categories:
            labels = custom_labels
        else:
            print('Custom label list length not matching number of desired categories! Default numbering will be used instead')
            labels = [str(i) for i in range(1, n_categories+1)]

    else:
        labels = [str(i) for i in range(1, n_categories+1)]

    for i in range(len(labels)):
        actual_level = {labels[i]: [values[i], values[i+1]]}
        categories.update(actual_level)

    return categories


def estimate_formula(formula: str, psyling_estimates: pd.DataFrame,
                     downgrade_factor: float = 0.5, sigmoid: bool = False,
                     multilevel: bool = False, n_categories: int = 4,
                     custom_labels: str = None,thresholds: typing.List[float
                                                                       ] = None) -> float:
    """
    Translates formula to score estimate from the datatable of feature scores

    :formula: string formula, e.g "A plus B minus C" [plus and minus are used]
    :psyling_estimates: DataFrame of feature estimates
    :downgrade_factor: custom, manual factor for weighting
    :sigmoid: Boolean, should the raw scoree be transformed with sigmoid function
    It's generally a good idea to turn this option on.
    :multilevel: Boolean, if True, the function creates labels and scores
    :n_categories: integer, ranges between 4 and 7, indicates the number of custom categories
    :custom_labels: list of string names of custom category names
    :returns: the score estimate

    """
    components = parse_formula(formula)
    components = [c for c in components if c != '']
    operators = ['plus', 'minus']
    score = 0

    for c in range(len(components)):
        if components[c] not in operators:
            if components[c] != '':
                estimate = list(psyling_estimates[components[c]])[0]
                if components[c-1] == 'minus':
                    estimate = estimate * (-1)
                score = score + estimate
                print(components[c])
                print(score)

    if sigmoid == False:
        # rescale the score
        rescaled_score = rescale(
            score,
            formula=formula,
            downgrade_factor=downgrade_factor)

        if rescaled_score > 1:
            rescaled_score = 1
        elif rescaled_score < 0:
            rescaled_score = 0

    elif sigmoid == True:
        rescaled_score = sigmoid_f(score)
        if multilevel == True and n_categories is not None:
            categories = yield_categories(
                n_categories, custom_labels=custom_labels,thresholds=thresholds)
            match = [
                rescaled_score >= values[0] and rescaled_score < values[1]
                for values in categories.values()]
            rescaled_score = {"label": str([[key for key in categories.keys()][i] for i in range(
                len(categories)) if match[i]][0]), "score":rescaled_score}

    return rescaled_score


def log_mlmodel(
        url: str, method: str, train_texts: typing.List[str],
        train_y: typing.List[str]) -> None:
    """
    Log training data change events on the server

    :url: string of the URL of the request
    :method: string of the HTTP request method
    :train_texts: string list of training texts used in the event
    :train_y: string list of response labels
    :returns: saves the log
    """
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    with open('app.log', 'a') as f:
        f.write('\n')
        entry = 'Log on ' + str(now) + ' at ' + url + " with " + method + '\n'
        stats = 'Number of training records is ' + str(len(train_texts)) + '\n'
        # urgent_count = 'urgent - not_urgent records: [' + str(
        #     sum(train_y == "urgent")) + ':' + str(sum(train_y == "not_urgent")) + ']\n'
        # record_match = len(train_y) == sum(
        #     train_y == 'urgent') + sum(train_y == 'not_urgent')
        f.write(entry)
        f.write(stats)
        # f.write(urgent_count)
        # f.write('Record matching [data integrity]: ' + str(record_match))
        f.close


def get_objects(text, nlp=None):
    doc = nlp(text.strip())
    objects = [token.pos_ for token in doc]
    return objects


def is_verb(items):
    verb = ['AUX', 'VERB']
    eval = []
    for v in verb:
        eval.append(v in items)
    return sum(eval) > 0


def tokenize_to_sentences(
        text: str, separators: typing.List[str] = []) -> typing.List[str]:
    """
    Tokenizes sentences preserving any punctuation
    :text: text input
    :separators: optional list of string separators for sub-sentence tokenization
    :returns: string list of sentences

    """

    if separators != []:
        pattern = separators
    else:
        pattern = "([?!.]) "

    if pattern == "([?!.]) ":
        split_text = re.split(pattern, text)
        split_text = [token for token in split_text if token != '']
        divs = ['.', '?', '!']
        res_text = []

        separators = [token in divs for token in split_text]
        separators_index = [i for i in range(len(separators)) if separators[i]]
        sentence_index = [sep - 1 for sep in separators_index]
        end_token_index = len(split_text) - 1

        for s in sentence_index:
            actual_sentence = split_text[s] + split_text[s+1]
            res_text.append(actual_sentence)

        res_text.append(split_text[end_token_index])
        res_text = [token for token in res_text if token not in divs]

    elif isinstance(pattern, str) and pattern != "([?!.]) ":
        mod_pattern = pattern.strip('( | )')
        mod_pattern = mod_pattern.replace("(", "")
        mod_pattern = mod_pattern.replace(")", "")
        split_text = re.split(mod_pattern, text)
        original_split = [t for t in re.split(pattern, text) if t]
        lengths = [get_word_count(item) for item in split_text]
        splits = []
        for i in range(len(lengths)):
            if i == 0:
                splits.append(lengths[i])
            else:
                splits.append(lengths[i] + splits[i-1])
        objects = get_objects(text, nlp=nlp)
        objects = [obj for obj in objects if obj != 'PUNCT']
        objects_split = []
        for s in range(len(splits)):
            actual_split = splits[s]
            if s == 0:
                objects_split.append(objects[:actual_split])
            else:
                previous_split = splits[s-1]
                objects_split.append(objects[previous_split:actual_split])
        verbs = [is_verb(item) for item in objects_split]
        verb_rate = [sum(verbs[i:i+2]) / 2 for i in range(len(verbs))]

        res_text = [""]
        for rate in range(len(verb_rate)):
            actual_rate = verb_rate[rate]

            if actual_rate > 0.5:
                if original_split[0].strip() in separators:
                    first_token = original_split[0] + original_split[1]
                    res_text.append(first_token)
                    [original_split.remove(original_split[0])
                     for i in range(2)]
                else:
                    first_token = original_split[0]
                    second_token = original_split[1] + original_split[2]
                    res_text.append(first_token)
                    res_text.append(second_token)
                    [original_split.remove(original_split[0])
                     for i in range(3)]
            else:
                if len(original_split) > 0:
                    res_text[len(
                        res_text) - 1] = res_text[len(res_text) - 1] + original_split[0]
                    original_split.remove(original_split[0])
                else:
                    pass

        if len(original_split) > 0:
            res_text[len(
                res_text)-1] = res_text[len(res_text)-1] + "".join(original_split)

    elif isinstance(pattern, list):
        sentences = tokenize_to_sentences(text)
        pattern = ['( '+item.lower()+' )' for item in pattern]
        pattern = '|'.join(pattern)
        tokenized_text = [
                          tokenize_to_sentences(
                              sentence, separators=pattern)
                          for sentence in sentences]
        tokenized_text = [s for item in tokenized_text for s in item if s]
        res_text = []
        separators = [token in pattern for token in tokenized_text]
        separators_index = [i for i in range(len(separators)) if separators[i]]
        sentence_tokens = [i == False for i in separators]
        sentence_index = [
            i for i in range(
                len(sentence_tokens)) if sentence_tokens[i]]
        end_token_index = len(tokenized_text) - 1
        index = {i: 'sentence' for i in sentence_index}
        for s in separators_index:
            index[s] = "separator"

        indices = [0]
        [indices.append(i) for i in range(len(index.keys()))
         if i > 0 and index[i-1] != "separator"]
        for i in indices:
            actual_token = index[i]
            if actual_token == 'separator':
                line = tokenized_text[i] + ' ' + tokenized_text[i+1]
            else:
                line = tokenized_text[i]
            res_text.append(line)

    return res_text


def create_vocabulary(
        name: str, tokens: typing.List[str],
        client: str = 's3') -> None:
    """
    Creates new custom vocabulary, saves it locally, and pushes it to S3
    :name: name of the new custom vocabulary
    :tokens: string list of tokens
    :client: string name of cloud client, one of ['s3', 'azure']
    """
    # a function to create and store (locally and on S3) the actual custom
    # vocabularies
    # !!! NOT the same as the function from psyling by the exact same name, it's a
    # modified version of that!!!
    # tokens is expected to be a List()
    filename = 'vocabularies/'+name+'.vocab'

    # create Pickle object locally
    with open(filename, 'wb') as f:
        pickle.dump(tokens, f)

    # push object to S3
    upload_object(client=client, file_name=filename)


def load_vocabulary(name: str, client: str = 's3') -> typing.List[str]:
    """
    Takes the name of a custom vocabulary, downloads it from S3, stores locally
    and returns it as a list
    :name: name of the custom vocabulary
    :client: string name of cloud client, one of ['s3', 'azure']
    """

    # loads vocabulary from S3
    # returns the vocab as list
    # NOT the same as the psyling.load_vocabulary()
    filename = 'vocabularies/'+name+'.vocab'

    download_object(client=client, file_name=filename)

    with open(filename, 'rb') as f:
        result_list = pickle.load(f)
    result_list.sort()

    return result_list


def change_vocabulary(
        name: str, new_tokens: typing.List[str],
        method: str = 'add', client='s3') -> None:
    """
    Changes existing vocabulary. The adjusted vocabulary is saved locally,
    then pushed to s3.
    :name: name of the custom vocabulary
    :new_tokens: list of tokens to be added to or removed from the vocabulary
    :method: string name of method, either 'add' or 'remove'or 'rename'
    :client: string name of cloud client, one of ['s3', 'azure']
    """
    # loads, changes and saves vocabulary
    # method can be one of [add, remove]
    # new tokens are expected to be presented to the function as List() of
    # strings
    filename = 'vocabularies/'+name+'.vocab'

    vocabulary = load_vocabulary(name)
    if method == 'add':
        vocabulary.extend(new_tokens)
        vocabulary.sort()
        vocabulary = list(set(vocabulary))

    if method == 'remove':
        [vocabulary.remove(token)
         for token in new_tokens if token in vocabulary]
        vocabulary = list(set(vocabulary))

    if method == 'rename':
        # rename vocabulary
        old_name = name
        new_name = str(new_tokens[0])
        filename = 'vocabularies/' + new_name + '.vocab'

        # adjust existing formulas with new vocab name
        existing_formulas = [
            "formulas/" + item
            for item in list_files(
                client=client, container_name="formulas")]

        for formula in existing_formulas:
            download_object(client=client, file_name=formula)

        for formula in existing_formulas:
            with open(formula, 'rb') as f:
                existing_formula = pickle.load(f)
                if existing_formula.find(old_name) > -1:
                    existing_formula = existing_formula.replace(
                        old_name, new_name)
                    with open(formula, 'wb') as f:
                        pickle.dump(existing_formula, f)
                    upload_object(client=client, file_name=formula)

    with open(filename, 'wb') as f:
        pickle.dump(vocabulary, f)
    upload_object(client=client, file_name=filename)


def render_formula(formula_name: str, client: str = 's3') -> dict:
    """
    Takes a fomula name and renders descriptives about the formula
    :formula_name: name of the formula
    :client: string name of cloud client, one of ['s3', 'azure']
    :returns: dictionary of formula details
    """
    # a function to render formulas to the GET request of /formulas

    filename = formula_name

    # get actual formula from S3
    if os.path.exists(filename):
        pass
    else:
        download_object(client=client, file_name=filename)

    # render return dict()
    with open(filename, 'rb') as f:
        formula = pickle.load(f)
    components = [item for item in parse_formula(
        formula) if item not in ['minus', 'plus']]
    full_formula = parse_formula(formula)
    count_of_components = len(components)
    positive_factors = [full_formula[0]]
    negative_factors = []

    for item in range(2, len(full_formula), 1):
        actual_element = [full_formula[item]]
        if full_formula[item - 1] == "plus":
            positive_factors.extend(actual_element)
        elif full_formula[item - 1] == "minus":
            negative_factors.extend(actual_element)

    positives = " + " + " + ".join(positive_factors)
    negatives = " - " + " - ".join(negative_factors)
    formula_short_name = formula_name.split("/")[1].split('.')[0]

    result = {
        'formula_estimation_name': formula_short_name,
        'length': count_of_components,
        'positive_factors': positives,
        'negative_factors': negatives
    }
    return result


def change_formula(formula_name: str, sign: str,
                   component: str, method: str, client: str = 's3') -> None:
    """
    Adjusts [extends] formulas with new elements or drops components from formulas

    :formula_name: name of the formula to be changed
    :sign: string representation of the sign of the new component to be added, one of [plus, minus]
    :component: string name of component to be added/removed
    :method: add a new component or remove and existing one; one of [add, remove]
    :client: string name of cloud client, one of ['s3', 'azure']
    """
    # load formula
    filename = 'formulas/' + formula_name + '.formula'
    download_object(client=client, file_name=filename)
    with open(filename, 'rb') as f:
        formula = pickle.load(f)

    # make change
    if method == "add":
        new_formula = formula + " " + sign + " " + component

    if method == "remove":
        string_to_remove = " " + sign + " " + component
        new_formula = formula.replace(string_to_remove, "")

    # save and store formula, push to S3
    with open(filename, 'wb') as f:
        pickle.dump(new_formula, f)
    upload_object(client=client, file_name=filename)
