import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses

    # Iterate over test sets
    for i in range(0, len(test_set.get_all_Xlengths())):
        sequence, length = test_set.get_item_Xlengths(i)
        log_likelihoods = {}

        # Score each model, and store the log likelihood
        for word, model in models.items():
            try:
                score = model.score(sequence, length)
                log_likelihoods[word] = score

            except:
                # If we can't score it, set the log likelihood to infinity
                log_likelihoods[word] = float("-inf")
                continue

        # Store log likelihoods for each test set
        probabilities.append(log_likelihoods)

        # Append max for each test set
        guesses.append(max(log_likelihoods, key=log_likelihoods.get))

    return probabilities, guesses
