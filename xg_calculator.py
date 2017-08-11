from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.special import expit

__author__ = 'Andrey Krivonogov <krivonogov.andrey@gmail.com>'

YARD = 0.9144
REG = 10e-7


def check_shot_in_pitch(x, y):
    if not (0 <= x <= 68 and 0 <= y <= 105):
        raise ValueError('Point {} is out of the pitch'.format((x, y)))


# noinspection PyTypeChecker
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
    def eval(self, **kwargs):
        """
        :param kwargs: should contain x meters from right (looking from the center of the pitch) touchline and
                                      y meters from goal line
        :return: expected goal result
        """
        pass

    def _relative_angle(self, x, y):
        """
        Angle from (x, y) to the nearest post divided by pi / 2 as described by Caley
        >>> xg = CaleyXGCalculator()
        >>> xg._relative_angle(x=44.948, y=1.392857)
        0.12021884320255356
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

    def _is_in_high_values_zone(self, x, y):
        return y < 5.5 and (30.34 < x < 37.66 or (30.34 - 5.5 < x < 37.66 + 5.5 and self._relative_angle(x, y) > 0.75))


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
            return {'1': -0.61, 'distance': -0.09, 'inverse_distance': 7.4, 'angle': 1.04, 'big_chance': 1.1,
                    'following_error': 0.67, ('inverse_distance', 'inverse_angle'): -3.2}
        elif direct_free_kick:
            # Shots from Direct Free Kicks
            return {'1': -3.84, 'distance': -0.1, 'inverse_distance': 98.7, 'inverse_angle': 3.54,
                    ('inverse_distance', 'inverse_angle'): -91.1}
        elif cross:
            if header:
                # Headed Shots Assisted by Crosses
                return {'1': -2.88, 'distance': -0.21, 'angle': 2.13, 'inverse_assist_distance': 4.31,
                        'assist_angle': 0.46, 'fastbreak': 0.2, 'counterattack': 0.11, 'set_piece': 0.12,
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
            return {'1': -3.85, 'distance': -0.1, 'inverse_distance': 2.56, 'angle': 1.94,
                    'throughball_assist': 0.51, 'fastbreak': 0.44, 'counterattack': 0.26, 'rebound': 0.7,
                    'established_possession': 0.44, 'otherbodypart': 1.14, 'big_chance': 1.3, 'following_error': 1.1,
                    'EPL': -0.29, 'LaLiga': -0.24, 'SerieA': -0.26}
        else:
            # Regular Shots
            # XXX: we use EPL and LaLiga coefs equal to 0.05 instead of -0.1 and -0.09 in order not to underestimate
            # results in these leagues
            return {'1': -3.19, 'distance': -0.095, 'inverse_distance': 3.18, 'angle': 1.88,
                    'inverse_angle': 0.24, ('inverse_distance', 'inverse_angle'): -2.09, 'throughball_assist': 0.45,
                    'throughball_2nd_assist': 0.64, 'assist_across_face': 0.31, 'cutback_assist': -0.15,
                    'inverse_assist_distance': 2.18, 'assist_angle': 0.12, 'fastbreak': 0.23, 'counterattack': 0.18,
                    'established_possession': 0.09, 'following_corner': -0.18, 'big_chance': 1.2,
                    'following_error': 1.1, 'following_dribble': 0.39, 'dribble_distance': 0.14, 'rebound': 0.37,
                    'game_state_sgn': 0.03, 'Bundesliga': 0.07, 'EPL': 0.05, 'LaLiga': 0.05, 'SerieA': -0.07}

    def eval(self, **kwargs):
        """
        Common kwargs:  'header', 'dribble_goalkeeper', 'cross', 'direct_free_kick', 'big_chance', 'following_error'
        League specific True/False kwargs: 'Bundesliga', 'EPL', 'LaLiga', 'SerieA'
        Assist kwargs for crosses and regular shots: 'assist_point'
        Only header related kwargs: 'otherbodypart', 'set_piece'
        Only cross related kwargs: 'corner'
        Speed related kwargs: 'fastbreak', 'counterattack'
        Other kwargs:   'throughball_assist', 'throughball_2nd_assist', 'cutback_assist',
                        'rebound', 'established_possession', 'game_state',
                        'following_corner', 'following_dribble', 'start_run_point'
        >>> xg = CaleyXGCalculator()
        >>> xg.eval(x=34, y=11, big_chance=True, throughball_assist=True, EPL=True, assist_x=34, assist_y=18)
        0.45227739927975769
        >>> xg.eval(x=37, y=20, direct_free_kick=True, LaLiga=True)
        0.10529256313817077
        >>> xg.eval(x=34, y=11)
        0.10695610254229779
        >>> xg.eval(x=34, y=11, assist_x=37, assist_y=25)
        0.12752188916686916
        >>> xg.eval(x=32, y=0.1)
        0.70504634688503887
        >>> xg.eval(x=30, y=7)
        0.15264470289986498
        >>> xg.eval(x=24.4, y=0.5, cross=True)
        0.027634985220878282
        >>> xg.eval(x=28.7, y=0.1, header=True) < 1e-5
        True
        >>> xg.eval(x=2, y=2) < 1e-5
        True
        >>> xg.eval(x=42, y=1) < 1e-5
        True
        >>> xg.eval(x=2, y=1, header=True) < 1e-3
        True
        >>> xg.eval(x=44.948, y=1.392857)
        2.9305605013432191e-05
        >>> xg.eval(x=33.660, y=11.785714)
        0.098866030922461545
        """
        x = kwargs['x']
        y = kwargs['y']
        check_shot_in_pitch(x, y)
        if not kwargs.get('dribble_goalkeeper', False) and y < 5.5 and 30.34 < x < 37.66:
            y = max(0.5, y)
        start_run_x, start_run_y = kwargs.pop('start_run_x', None), kwargs.pop('start_run_y', None)
        assist_x, assist_y = kwargs.pop('assist_x', None), kwargs.pop('assist_y', None)
        verbose = kwargs.pop('verbose', False)

        body_part = kwargs.pop('body_part', 'RightFoot')
        kwargs['header'] = body_part == 'header' or body_part == 'otherbodypart'
        kwargs['otherbodypart'] = body_part == 'otherbodypart'

        kwargs['game_state_sgn'] = np.sign(kwargs.get('game_state', 0))
        kwargs[kwargs.pop('league', None)] = True
        kwargs[kwargs.pop('pass_type', None)] = True
        kwargs[kwargs.pop('attack_type', 'open_play')] = True
        kwargs['counterattack'] = kwargs.get('counterattack', False) or kwargs.get('fastbreak', False)

        cross_or_header = kwargs.get('cross', False) or kwargs['header']

        shot_params = {k: (float(v) if not isinstance(v, tuple) else v) for k, v in kwargs.iteritems()}
        if cross_or_header:
            shot_params['distance'] = self._distance(x, y, self.goal_center, yards=True)
        else:
            shot_params['distance'] = y / YARD

        shot_params['angle'] = self._relative_angle(x, y)

        if start_run_x is not None and start_run_y is not None and not cross_or_header:
            start_distance = max(0.5, start_run_y) / YARD
            shot_params['dribble_distance'] = (start_distance - shot_params['distance']) / start_distance

        if assist_x is not None and assist_y is not None:
            assist_x = float(assist_x)
            assist_y = float(assist_y)
            shot_params['assist_distance'] = self._distance(assist_x, assist_y, self.goal_center, yards=True)
            shot_params['assist_angle'] = self._relative_angle(assist_x, assist_y)

        return log_reg_dot(self._beta(kwargs.get('dribble_goalkeeper', False),
                                      kwargs.get('direct_free_kick', False),
                                      kwargs.get('cross', False),
                                      kwargs.get('header', False)),
                           shot_params, verbose=verbose)
