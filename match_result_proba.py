import itertools

import numpy as np

__author__ = 'Andrey Krivonogov <krivonogov.andrey@gmail.com>'

INCREDIBLY_BIG_NUMBER_OF_GOALS = 10


def number_of_goals_probabilities(p_shots):
    """
    :param p_shots: array or list: probabilities for each shot to scored
    :return: array where at position k we have probability that exactly k shots will be scored
    >>> number_of_goals_probabilities([0.1, 0.1])
    array([ 0.81,  0.18,  0.01])
    >>> number_of_goals_probabilities([0.8, 0.2, 0.2])
    array([ 0.128,  0.576,  0.264,  0.032])
    >>> number_of_goals_probabilities([0.1] * 15).shape
    (11,)
    >>> p_shots = np.random.exponential(0.1, 10)
    >>> np.sum(number_of_goals_probabilities(p_shots) * np.arange(11)) - p_shots.sum() < 1e-3
    True
    """
    p_shots = np.sort(np.asarray(p_shots, dtype=np.float64))
    p_compl = 1.0 - p_shots

    all_idx = set(range(len(p_shots)))

    p_goals = np.zeros(min(len(p_shots), INCREDIBLY_BIG_NUMBER_OF_GOALS) + 1)

    for k in xrange(len(p_goals)):
        if k > 5:
            # max_proba = binom(len(p_shots), k) * np.prod(p_shots[-k:]) * np.prod(1 - p_shots[:-k])
            binom_nom = len(p_shots) - np.arange(k)
            binom_den = k - np.arange(k)
            max_proba = np.prod(binom_nom * p_shots[-k:] / binom_den) * np.prod(p_compl[:-k])
            # we suppose that every next maximal probability is at least 1.5 times smaller than current one
            # this can be violated only if there is a lot of shots with xg >= 0.5 (but in this cases team should have
            # much bigger chances to score more goals than opponent, so the influence on winning probability
            # will be small)
            max_proba_for_greater_k = max_proba * 3

            if max_proba_for_greater_k < 1e-3:
                return p_goals

        for idx_scored in itertools.combinations(range(len(p_shots)), k):
            scored_proba = np.prod(p_shots[list(idx_scored)])
            idx_missed = list(all_idx - set(idx_scored))
            missed_proba = np.prod(p_compl[idx_missed])

            p_goals[k] += scored_proba * missed_proba

    return p_goals


def match_result_probabilities(p_shots_home, p_shots_away):
    """
    :param p_shots_home: array or list: probabilities for each home team's shot to scored
    :param p_shots_away: array or list: probabilities for each away team's shot to scored
    :return: array of size 3 with probabilities of (home win, draw, away win)
    >>> match_result_probabilities([0.4], [0.3])
    array([ 0.28,  0.54,  0.18])
    >>> match_result_probabilities([0.1, 0.1], [0.05, 0.15])
    array([ 0.1553,  0.6875,  0.1573])
    >>> match_result_probabilities([0.1, 0.1, 0.1], [0.2, 0.1])
    array([ 0.2024,  0.5886,  0.209 ])
    """
    p_home = number_of_goals_probabilities(p_shots_home)
    p_away = number_of_goals_probabilities(p_shots_away)

    p_home_win = 0.
    p_draw = 0.
    p_away_win = 0.
    for goals_home in xrange(len(p_home)):
        for goals_away in xrange(len(p_away)):
            p = p_home[goals_home] * p_away[goals_away]
            if goals_home > goals_away:
                p_home_win += p
            elif goals_home < goals_away:
                p_away_win += p
            else:
                p_draw += p

    return np.round([p_home_win, p_draw, p_away_win], 4)
