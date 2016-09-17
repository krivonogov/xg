from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.special import expit

__author__ = 'Andrey Krivonogov <krivonogov.andrey@gmail.com>'

YARD = 0.9144
REG = 10e-7
CUTBACK_THRESH_IN_METERS = 4


def check_shot_in_pitch(x, y):
    if not (0 <= x <= 68 and 0 <= y <= 105):
        raise ValueError('Point {} is out of the pitch'.format((x, y)))


def log_reg_dot(beta_dict, shot_params, verbose=False):
    """
    :param beta_dict: dict with beta coefficients
    :param shot_params: dict with data point
    :param verbose: verbose calculation coefficient by coefficient
    :return: sigmoid of dot product beta * data point
    >>> log_reg_dot({'1': 1, 'a': -1, 'b': -0.5, 'c': 1000}, {'a': 0.5, 'b': 1, 'foo': 2000})
    0.5
    """
    res = 0.
    if verbose:
        print 'Key', 'beta', 'X', 'beta*X', 'res'

    for k in beta_dict.keys():
        if k in shot_params:
            res += beta_dict[k] * shot_params[k]
            if verbose:
                print k, beta_dict[k], shot_params[k], beta_dict[k] * shot_params[k], res
        elif isinstance(k, tuple):
            arguments = []
            for key in k:
                if key in shot_params:
                    arg = shot_params[key]
                elif key.startswith('inverse_'):
                    k_inv = key.replace('inverse_', '')
                    if k_inv in shot_params:
                        arg = 1.0 / (shot_params[k_inv] + REG)
                    else:
                        arg = 0.
                else:
                    arg = 0.
                arguments.append(arg)

            res += beta_dict[k] * np.prod(arguments)
            if verbose:
                print k, beta_dict[k], arguments, beta_dict[k] * np.prod(arguments), res
        elif k.startswith('inverse_'):
            k_inv = k.replace('inverse_', '')
            if k_inv in shot_params:
                res += beta_dict[k] / (shot_params[k_inv] + REG)
                if verbose:
                    print 'Inv:', k, beta_dict[k], shot_params[k_inv], beta_dict[k] / (shot_params[k_inv] + REG), res

    res += beta_dict['1']
    if verbose:
        print 'Dot result:', res

    # Player adjustment for Caley's XG
    player_adjustment = shot_params.get('player_adjustment', 1.)
    if res < 0:
        player_adjustment = 1. / player_adjustment

    if verbose:
        if abs(player_adjustment - 1) > 1e-6:
            print 'Adjusted result:', res * player_adjustment
        print 'Logreg result:', expit(res * player_adjustment)

    return expit(res * player_adjustment)


def if_assist_across_face(x, y, assist_x, assist_y):
    y_box = 5.5
    x_min_box = 30.34 - 5.5
    x_max_box = 37.66 + 5.5

    if (x_min_box < x < x_max_box and y < y_box) or (x_min_box < assist_x < x_max_box and assist_y < y_box):
        # one of the points is in the box
        return True

    if y >= y_box and assist_y >= y_box:
        # both points are above y = y_box
        return False

    if ((x - x_min_box) * (assist_x - x_min_box) >= 0) and ((x - x_max_box) * (assist_x - x_max_box) >= 0):
        # both points are in the same coridor
        return False

    # other cases are treated using
    # http://stackoverflow.com/questions/4977491/determining-if-two-line-segments-intersect/4977569#4977569
    # we check whether passing line intersects with one of the lines of box
    for x0, y0, x1, y1 in [(x_min_box, 0, x_min_box, y_box),  # x = x_min_box
                           (x_min_box, y_box, x_max_box, y_box),  # y = y_box
                           (x_max_box, 0, x_max_box, y_box)]:  # x = x_max_box
        det = (x1 - x0) * (y - assist_y) - (x - assist_x) * (y1 - y0)
        if abs(det) < 1e-6:
            continue
        s = (assist_x - x0) * (y - assist_y) - (assist_y - y0) * (x - assist_x)
        s /= det
        t = (assist_x - x0) * (y1 - y0) - (assist_y - y0) * (x1 - x0)
        t /= det
        if 0 < s < 1 and 0 < t < 1:
            return True
    return False


class XGCalculator(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        """
        We work in coordinates on a pitch with measure in meters ie
        pitch is bounded in 0 < x < 68, 0 < y < 105, with a goal posts situated at (30.34, 0) and (37.66, 0)
        """
        self.left_post = np.array([30.34, 0], np.float64)
        self.right_post = np.array([37.66, 0], np.float64)
        self.goal_center = np.array([34, 0], np.float64)
        self.goal_width = self.right_post[0] - self.left_post[0]

    @abstractmethod
    def eval(self, x, y, **kwargs):
        """
        :param x: meters from right (looking from the center of the pitch) touchline
        :param y: meters from goal line
        :return: expected goal result
        """
        pass

    def _relative_angle(self, x, y):
        """
        Angle from (x, y) to the nearest post divided by pi / 2 as described by Caley
        """
        if self.left_post[0] <= x <= self.right_post[0]:
            return 1.
        elif x < self.left_post[0]:
            x_to_post = self.left_post[0] - x
        else:
            x_to_post = x - self.right_post[0]

        return 2 * np.arctan(y / x_to_post) / np.pi

    @staticmethod
    def _distance(x, y, point, yards=False):
        """
        Returns distance from (x, y) to point
        :param yards: if True distance is returned in yards
        """
        return np.linalg.norm(np.asarray([x, y]) - np.asarray(point)) / (YARD if yards else 1)

    def _angle_between_posts(self, x, y):
        """
        Returns angle made by position of shot and goalposts by cosine rule
        """
        dist_left = self._distance(x, y, self.left_post)
        dist_right = self._distance(x, y, self.right_post)

        cos_angle = (dist_left ** 2 + dist_right ** 2 - self.goal_width ** 2) / (2 * dist_left * dist_right)
        return np.arccos(cos_angle)


class SimpleXGCalculator(XGCalculator):
    """
    XG model with beta coefficients used in http://torvaney.github.io/xG.html
    """
    def __init__(self):
        super(SimpleXGCalculator, self).__init__()
        # order of features:
        # 0     intercept
        # 1:4   angle, goal_dist, header
        # 4:7   angle * goal_dist, angle * header, goal_dist * header
        # 7     angle * goal_dist * header
        self._beta = {'1': -1.745598, 'angle': 1.338737, 'distance': -0.110384, 'header': 0.646730,
                      ('angle', 'distance'): 0.168798, ('angle', 'header'): -0.424885,
                      ('distance', 'header'): -0.134178, ('angle', 'distance', 'header'): -0.055093}

    def eval(self, x, y, **kwargs):
        """
        Kwargs: header, verbose
        >>> xg = SimpleXGCalculator()
        >>> xg.eval(34, 11)
        0.28761592428370542
        >>> xg.eval(37, 20)
        0.09166111179904099
        """
        check_shot_in_pitch(x, y)
        header = int(kwargs.get('header', False))
        verbose = kwargs.pop('verbose', False)

        angle = self._angle_between_posts(x, y)

        # distance to the centre of the goal
        goal_dist = self._distance(x, y, self.goal_center)

        return log_reg_dot(self._beta, {'angle': angle, 'distance': goal_dist, 'header': header}, verbose=verbose)


class CaleyXGCalculator(XGCalculator):
    """
    XG model introduced by Caley in
    http://cartilagefreecaptain.sbnation.com/2015/10/19/9295905/premier-league-projections-and-new-expected-goals
    """
    def __init__(self):
        super(CaleyXGCalculator, self).__init__()

    @staticmethod
    def _beta(dribble_goal, direct_free_kick, cross, header):
        if dribble_goal:
            # Shots Following a Dribble of the Keeper
            return {'1': -0.61, 'distance': -0.09, 'inverse_distance': 7.4, 'angle': 1.04, 'big_change': 1.1,
                    'following_error': 0.67, ('inverse_distance', 'inverse_angle'): -3.2}
        elif direct_free_kick:
            # Shots from Direct Free Kicks
            # XXX: We use coeff 3.34 for inverse_angle instead of 3.54 to penalize free kicks at sharp angles
            return {'1': -3.84, 'distance': -0.1, 'inverse_distance': 98.7, 'inverse_angle': 3.34,
                    ('inverse_distance', 'inverse_angle'): -91.1}
        elif cross:
            if header:
                # Headed Shots Assisted by Crosses
                return {'1': -2.88, 'distance': -0.21, 'relative_angle': 2.13, 'inverse_assist_distance': 4.31,
                        'assist_angle': 0.46, 'fastbreak': 0.2, 'counterattack': 0.11, 'set_play': 0.12,
                        'corner': -0.24, 'otherbodypart': -0.18, 'big_chance': 1.2, 'following_error': 1.1,
                        'EPL': 0.18, 'LaLiga': 0.15}
            else:
                # Non-Headed Shots Assisted by Crosses
                return {'1': -2.8, 'distance': -0.11, 'inverse_distance': 3.52, 'angle': 1.14,
                        'assist_across_face': 0.14, 'inverse_assist_distance': 6.94, 'assist_angle': 0.59,
                        'corner': -0.12, 'fastbreak': 0.24, 'counterattack': 0.11, 'big_chance': 1.25,
                        'following_error': 1.1, 'EPL': -0.2}
        elif header:
            # Headed Shots Not Assisted by Crosses
            return {'1': -3.85, 'distance': -0.1, 'inverse_distance': 2.56, 'relative_angle': 1.94,
                    'throughball_assist': 0.51, 'fastbreak': 0.44, 'counterattack': 0.26, 'rebound': 0.7,
                    'established_possession': 0.44, 'otherbodypart': 1.14, 'big_chance': 1.3, 'following_error': 1.1,
                    'EPL': -0.29, 'LaLiga': -0.24, 'SerieA': -0.26}
        else:
            # Regular Shots
            # XXX: we use EPL and LaLiga coefs equal to 0.05 instead of -0.1 and -0.09 in order not to underestimate
            # results in these leagues
            return {'1': -3.19, 'distance': -0.095, 'inverse_distance': 3.18, 'relative_angle': 1.88,
                    'inverse_angle': 0.24, ('inverse_distance', 'inverse_angle'): -2.09, 'throughball_assist': 0.45,
                    'throughball_2nd_assist': 0.64, 'assist_across_face': 0.31, 'cutback_assist': -0.15,
                    'inverse_assist_distance': 2.18, 'assist_angle': 0.12, 'fastbreak': 0.23, 'counterattack': 0.18,
                    'established_possession': 0.09, 'following_corner': -0.18, 'big_chance': 1.2,
                    'following_error': 1.1, 'following_dribble': 0.39, 'dribble_distance': 0.14, 'rebound': 0.37,
                    'game_state': 0.03, 'Bundesliga': 0.07, 'EPL': 0.05, 'LaLiga': 0.05, 'SerieA': -0.07}

    def eval(self, x, y, **kwargs):
        """
        Common kwargs:  'header', 'dribble_goalkeeper', 'cross', 'direct_free_kick', 'big_chance', 'following_error'
        League specific True/False kwargs: 'Bundesliga', 'EPL', 'LaLiga', 'SerieA'
        Assist kwargs for crosses and regular shots: 'assist_point'
        Only header related kwargs: 'otherbodypart', 'set_play'
        Only cross related kwargs: 'corner'
        Speed related kwargs: 'fastbreak', 'counterattack'
        Other kwargs:   'throughball_assist', 'throughball_2nd_assist', 'cutback_assist',
                        'rebound', 'established_possession', 'game_state',
                        'following_corner', 'following_dribble', 'start_run_point'
        >>> xg = CaleyXGCalculator()
        >>> xg.eval(34, 11, big_chance=True, throughball_assist=True, EPL=True, assist_point=(34, 18))
        0.45227739927975757
        >>> xg.eval(37, 20, direct_free_kick=True, LaLiga=True)
        0.085640339701528026
        """
        check_shot_in_pitch(x, y)
        start_run_point = kwargs.pop('start_run_point', None)
        assist_point = kwargs.pop('assist_point', None)
        verbose = kwargs.pop('verbose', False)

        shot_params = {k: (float(v) if not isinstance(v, tuple) else v) for k, v in kwargs.iteritems()}
        shot_params['distance'] = self._distance(x, y, self.goal_center, yards=True)
        shot_params['relative_angle'] = self._relative_angle(x, y)
        shot_params['angle'] = shot_params['relative_angle']

        if start_run_point is not None:
            start_run_x, start_run_y = start_run_point
        else:
            start_run_x, start_run_y = x, y
        start_distance = self._distance(start_run_x, start_run_y, self.goal_center, yards=True)
        shot_params['dribble_distance'] = (start_distance - shot_params['distance']) / start_distance

        if assist_point is not None:
            assist_x, assist_y = assist_point
            shot_params['assist_distance'] = self._distance(assist_x, assist_y, self.goal_center, yards=True)
            shot_params['assist_angle'] = self._relative_angle(assist_x, assist_y)
            shot_params['assist_across_face'] = float(if_assist_across_face(x, y, assist_x, assist_y))
            shot_params['cutback_assist'] = float((y - assist_y) > CUTBACK_THRESH_IN_METERS)

        return log_reg_dot(self._beta(kwargs.get('dribble_goalkeeper', False),
                                      kwargs.get('direct_free_kick', False),
                                      kwargs.get('cross', False),
                                      kwargs.get('header', False)),
                           shot_params, verbose=verbose)
