"""Kanerva coding code is modified from code by Kris de Asis.
Plotting code is modified from code by Cam Linke.
Everything else by Niko Yasui.

Use Python 3.5 or higher.
"""
import asyncio
from collections import deque, ChainMap
import math
import multiprocessing as mp
import time
from functools import partial
from itertools import product, chain
from typing import (Callable, Iterable, List, NewType, Sequence, Tuple,
                    Dict, Union)

import numpy as np
import serial_asyncio
import visdom

# from kanerva import KanervaCoder


# TOOLS


def timer_factory(name, indents=0):
    def timing(f):
        stats = {'mean': 0,
                 'M2': 0,
                 'n': 0}
        indent = ''
        for _ in range(indents):
            indent += '\t'
        msg = indent + "{} took {:.4f}/{:.4f}/{:.2f}% current/mean/rsd seconds."

        def wrap(*args, **kwargs):
            start = time.time()
            out = f(*args, **kwargs)
            end = time.time()
            x = end - start
            old_mean = stats['mean']
            stats['n'] += 1
            stats['mean'] += (x - stats['mean']) / stats['n']
            stats['M2'] += (x - old_mean) * (x - stats['mean'])

            rsd = 100 * math.sqrt(stats['M2'] / stats['n']) / abs(stats['mean'])
            print(msg.format(name, x, stats['mean'], rsd))

            return out

        return wrap

    return timing


# SERVO FUNCTIONS

def write(writer, sid, instruction):
    # msg format: 0xff, 0xff, sid, msg length, instruction, checksum
    msg = [sid, len(instruction) + 1] + instruction
    checksum = ~sum(msg) % 256
    out = [0xff, 0xff] + msg + [checksum]
    writer.write(bytes(out))


async def send_msg(reader, writer, sid, instruction):
    write(writer, sid, instruction)
    return await get_return_packet(reader, sid)


async def get_return_packet(reader, sid):
    assert await reader.readexactly(2) == b'\xff\xff'  # check start bits
    assert ord(await reader.readexactly(1)) == sid  # check servo ID

    data_len = ord(await reader.readexactly(1))

    try:
        err = ord(await reader.readexactly(1))
        assert err == 0
    except AssertionError:
        raise RuntimeError(f"Error byte had non-zero value: {err}.")

    data = await reader.readexactly(data_len - 2)
    await reader.readexactly(1)  # checksum, doesn't seem to work properly

    return data


def parse_data(data):
    if len(data) > 8:
        data = data[36:44]

    position = (data[0] + data[1] * 256 - 512) * 5 * math.pi / 3069
    speed = (data[2] + data[3] * 256) * 5 * math.pi / 3069
    load = -1 if (data[5] >> 2 & 1) == 0 else 1
    # load = (data[5] & 3) * 256 + data[4] * (1 - 2 * bool(data[5] & 4))
    voltage = data[6] / 10
    temperature = data[7]

    return [position, speed, load, voltage, temperature]


def goal_instruction(angle):
    enc = int(round(angle * 3069 / math.pi / 5 + 512))
    hi, lo = enc // 256, enc % 256

    return [0x03,  # write
            0x1e,  # to goal position
            lo, hi]  # desired position


# REINFORCEMENT LEARNING

obsv = NewType('obsv', Sequence[float])
actn = NewType('action', Tuple[int])
dat = NewType('dat', Sequence[Sequence[float]])


class DiscretePolicy:
    def __init__(self,
                 actions: Sequence[actn],
                 actn_probs: Callable[..., Sequence[float]]):
        self.actions = actions
        self.actn_probs = actn_probs

    def __call__(self, *args, **kwargs) -> (actn, float):
        action_probabilities = self.actn_probs(*args, **kwargs)
        index = np.random.choice(a=len(self.actions),
                                 p=action_probabilities)

        return self.actions[index], action_probabilities[index]

    def prob(self, a: actn, *args, **kwargs) -> float:
        action_probabilities = self.actn_probs(*args, **kwargs)
        index = self.actions.index(a)
        return action_probabilities[index]


class RUPEE:
    def __init__(self,
                 n_features: int,
                 beta_0: float,
                 alpha_h_hat: float):
        # RUPEE, see Adam White's 2015 thesis p.132, 136, 138, 142
        self.alpha_h_hat = alpha_h_hat
        self.beta0 = beta_0
        self.h_hat = np.zeros(n_features)
        self.d_e = np.zeros(n_features)
        self.tau = 0

    def update(self,
               delta_e: np.ndarray=None,
               x: np.ndarray=None,
               **kwargs):
        del kwargs
        tau, h_hat, d_e, beta0_r = self.tau, self.h_hat, self.d_e, self.beta0

        # rupee
        tau = (1 - beta0_r) * tau + beta0_r
        h_hat[x] += self.alpha_h_hat * (delta_e[x] - h_hat[x].sum())
        d_e = ((1 - beta0_r / tau) * d_e + beta0_r / tau * delta_e)

        self.tau, self.h_hat, self.d_e, self.beta0 = tau, h_hat, d_e, beta0_r

    @property
    def value(self) -> float:
        return np.sqrt(np.abs(np.dot(self.h_hat, self.d_e)))


class UDE:
    def __init__(self, beta0: float):
        # UDE
        self.tau = 0
        self.beta0 = beta0
        self.ewma = 0
        self.sample_mean = 0
        self.n = 0
        self.sample_var = 0

    def update(self, delta: float=None, **kwargs):
        del kwargs
        tau, beta0 = self.tau, self.beta0

        # ude
        tau = (1 - beta0) * tau + beta0
        self.ewma = (1 - beta0 / tau) * self.ewma + (beta0 / tau) * delta
        old_mean = self.sample_mean
        self.n += 1
        self.sample_mean += (delta - self.sample_mean) / self.n
        var_sample = (delta - old_mean) * (delta - self.sample_mean)
        self.sample_var += (var_sample - self.sample_var) / self.n

        self.tau, self.beta0 = tau, beta0

    @property
    def value(self) -> float:
        stdev = np.sqrt(self.sample_var) + np.finfo(float).eps
        return np.abs(self.ewma / stdev)


Evaluator = Union[RUPEE, UDE]


class Learner:
    def __init__(self,
                 alpha: float,
                 gamma: Callable[[obsv], float],
                 cumulant: Callable[[np.ndarray, obsv, actn, np.ndarray, obsv],
                                    float],
                 lamda: float,
                 name: str):
        self.alpha = alpha * (1 - lamda)
        self.gamma = gamma
        self.lamda = lamda
        self.cumulant = cumulant
        self.name = name


# class WISTOGTDLearner(Learner):
#     def __init__(self,
#                  alpha: float,
#                  gamma: Callable[[obsv], float],
#                  cumulant: Callable[[np.ndarray, obsv, actn, np.ndarray, obsv],
#                                     float],
#                  lamda: float,
#                  target_policy: DiscretePolicy,
#                  n_features: int,
#                  name: str,
#                  evaluators: Dict[str, Evaluator]=None,
#                  alpha_h: float=None,
#                  *args, **kwargs):
#         del args, kwargs
#         super().__init__(alpha, gamma, cumulant, lamda, name)
#
#         self.alpha_h = self.alpha / 100 if alpha_h is None else alpha_h
#         self.w = np.zeros(n_features)
#         self.e = np.zeros(n_features)
#         self.h = np.zeros(n_features)
#         self.la = initial_action
#         self.target_policy = target_policy
#
#         self.eh = np.zeros(n_features)
#         self.old_w = np.zeros(n_features)
#
#         self.evaluators = {} if evaluators is None else evaluators
#         self.delta = None
#
#         # u0 = -self.alpha
#         # self.u = np.ones(n_features) * u0
#         # self.v = np.zeros(n_features)
#         # self.eta = self.alpha / 100
#         self.gamma_t = 1
#         self.rhop = 0
#
#     # def update_alpha(self, rho, x):
#     #     etaphi = np.ones_like(self.u)
#     #     etaphi[x] -= self.eta
#     #     self.u *= etaphi
#     #     self.u[x] += rho
#     #     self.u += (rho - 1) * self.gamma_t * self.lamda * etaphi * self.v
#     #     self.v *= self.gamma_t * self.lamda * rho * etaphi
#     #     self.v[x] += rho
#     #
#     #     self.alpha = np.zeros_like(self.u)
#     #     inds = self.u != 0
#     #     self.alpha[inds] = 1 / self.u[inds]
#
#     def update(self,
#                action: actn,
#                action_prob: float,
#                obs: obsv,
#                obsp: obsv,
#                x: np.ndarray,
#                xp: np.ndarray):
#
#         # for readability
#         w, e, h, lamda = self.w, self.e, self.h, self.lamda
#
#         # precalculations
#         gammap = self.gamma(obsp)
#         rho = self.target_policy.prob(w=w, x=x, obs=obs, a=action) / action_prob
#         reward = self.cumulant(x, obs, action, xp, obsp)
#         self.delta = reward + gammap * w[xp].sum() - w[x].sum()
#         assert rho == 1 or self.alpha_h > 0
#         # self.update_alpha(rho, x)
#         alpha = self.alpha
#
#         # gtd
#         rgl = rho * self.gamma_t * self.lamda
#         ed = rgl * self.e
#         ed[x] += 1
#         rap = np.zeros_like(self.e)
#         rap[x] += rho * alpha
#         self.e = rgl * (self.e - rho * rap * self.e[x].sum()) + rap
#         de = self.delta * self.e
#         glrp = self.gamma_t * self.lamda * self.rhop
#         self.eh = glrp * self.eh + self.alpha_h * (1 - glrp * self.eh[x].sum())
#
#         self.w += de + (self.e - rap) * (self.w - self.old_w)[x].sum()
#         self.w[xp] -= self.alpha * gammap * (1 - self.lamda) * np.dot(ed, h)
#         h += rho * self.delta * self.eh
#         h[x] -= self.alpha_h * self.h[x].sum()
#
#         # evaluators
#         ev_kwargs = {'delta_e': de, 'x': x, 'delta': self.delta}
#         for k, ev in self.evaluators.items():
#             ev.update(**ev_kwargs)
#
#         # reassign variables
#         self.e, self.h = e, h
#         self.gamma_t = gammap
#         self.rhop = rho
#         self.old_w = self.w
#
#     def predict(self, x: np.ndarray) -> float:
#         return self.w[x].sum()
#
#     def data(self,
#              x: np.ndarray,
#              obs: obsv,
#              action: actn,
#              xp: np.ndarray,
#              obsp: obsv) -> Dict[str, Dict[str, float]]:
#         return {self.name: {**{k: v.value for k, v in self.evaluators.items()},
#                             **{'cumul': self.cumulant(x, obs, action, xp, obsp),
#                                'pred': self.predict(x)},
#                             'tderr': self.delta}}


class GTDLearner(Learner):
    def __init__(self,
                 alpha: float,
                 gamma: Callable[[obsv], float],
                 cumulant: Callable[[np.ndarray, obsv, actn, np.ndarray, obsv],
                                    float],
                 lamda: float,
                 target_policy: DiscretePolicy,
                 n_features: int,
                 name: str,
                 evaluators: Dict[str, Evaluator]=None,
                 alpha_h: float=None,
                 *args, **kwargs):
        del args, kwargs
        super().__init__(alpha, gamma, cumulant, lamda, name)

        self.alpha_h = self.alpha / 100 if alpha_h is None else alpha_h
        self.w = np.zeros(n_features)
        self.e = np.zeros(n_features)
        self.h = np.zeros(n_features)
        self.la = initial_action
        self.target_policy = target_policy

        self.evaluators = {} if evaluators is None else evaluators
        self.delta = None

        self.gamma_t = 1
        self.rho = 1

    def update(self,
               action: actn,
               action_prob: float,
               obs: obsv,
               obsp: obsv,
               x: np.ndarray,
               xp: np.ndarray,
               ude: bool=False,
               rupee: bool=False):
        # for readability
        alpha, w, e, h, lamda = self.alpha, self.w, self.e, self.h, self.lamda

        # precalculations
        gammap = self.gamma(obsp)
        pi = self.target_policy.prob(w=w, x=x, obs=obs, a=action)
        self.rho = pi / action_prob
        reward = self.cumulant(x, obs, action, xp, obsp)
        self.delta = reward + gammap * w[xp].sum() - w[x].sum()
        assert self.rho == 1 or self.alpha_h > 0

        # gtd
        e *= self.rho * lamda * self.gamma(obs)
        e[x] += self.rho
        de = self.delta * e
        w += alpha * de
        w[xp] -= alpha * gammap * (1 - lamda) * np.dot(e, h)
        if rupee:
            w *= -1
        h += self.alpha_h * de
        h[x] -= self.alpha_h * h[x].sum()

        # evaluators
        ev_kwargs = {'delta_e': de,
                     'x': x,
                     'delta': 100 if ude else self.delta}
        for k, ev in self.evaluators.items():
            ev.update(**ev_kwargs)

    def predict(self, x: np.ndarray) -> float:
        return self.w[x].sum()

    def data(self,
             x: np.ndarray,
             obs: obsv,
             action: actn,
             xp: np.ndarray,
             obsp: obsv) -> Dict[str, Dict[str, float]]:
        return {self.name: {**{k: v.value for k, v in self.evaluators.items()},
                            **{'cumul': self.cumulant(x, obs, action, xp, obsp),
                               'pred': self.predict(x)},
                            'tderr': self.delta,
                            'rho': self.rho,
                            'gammap': self.gamma(obsp)}}


class Plot:
    def __init__(self,
                 parser: Callable[[Sequence[Sequence[float]]], Iterable[float]],
                 title: str,
                 legend: List[str],
                 ylab: str,
                 xlab: str='Time-step',
                 size: int=50):
        self.parser = parser
        self.title = title

        self.y_windows = [deque(size * [0], size)
                          for _ in range(len(legend))]
        self.predicted_window = deque(size * [0], size)
        self.x_window = deque([0 for _ in range(size)], size)

        self.moving_window = vis.line(
            X=np.asarray([0, 0]),
            Y=np.column_stack([[0, 0]] * len(legend)),
            opts=dict(
                showlegend=True,
                width=700,
                height=300,
                xlabel=xlab,
                ylabel=ylab,
                title=title,
                marginleft=30,
                marginright=30,
                marginbottom=80,
                margintop=30,
                legend=legend,
            ),
        )

        self.i = 0
        self.size = size
        self.y = [[] for _ in range(len(legend))]

    def update(self, data: dat):
        try:
            self.update_plot(*self.parser(data))
        except KeyError:
            print(data)
            raise

    def update_plot(self, *args: float):
        self.i += 1

        for y_list in self.y_windows, self.y:
            for y, a in zip(y_list, args):
                y.append(a)
        self.x_window.append(self.i)

        # Update windowed graph
        # vis.line(
        #     X=np.asarray(self.x_window),
        #     Y=np.column_stack(self.y_windows),
        #     win=self.moving_window,
        #     update='replace',
        #     opts=dict(showlegend=True)
        # )


def horde_plots(gvf_names: List[str], name: str) -> List[Plot]:
    return [Plot(parser=lambda data: [data[gname]['pred']
                                      for gname in gvf_names],
                 title=f'{name.capitalize()} predictions',
                 legend=gvf_names,
                 ylab='Predictions',
                 size=50),
            Plot(parser=lambda data: [data[gname]['rupee']
                                      for gname in gvf_names],
                 title=f'{name.capitalize()} RUPEEs',
                 legend=gvf_names,
                 ylab='RUPEE',
                 size=50),
            Plot(parser=lambda data: [data[gname]['ude']
                                      for gname in gvf_names],
                 title=f'{name.capitalize()} UDEs',
                 legend=gvf_names,
                 ylab='UDE',
                 size=100),
            Plot(parser=lambda data: [data[gname]['tderr']
                                      for gname in gvf_names],
                 title=f'{name.capitalize()} TD errors',
                 legend=gvf_names,
                 ylab='TD Error',
                 size=100),
            ]


class KanervaCoder:
    """Kanerva coding class modified from Kris De Asis's from
    https://github.com/MeepMoop/kanervacoding
    """
    def __init__(self, dims, ptypes, sparsity, limits, seed=None):
        assert len(limits) == dims

        np.random.seed(seed)
        self._n_dims = dims
        self._n_pts = ptypes
        self._k = int(round(sparsity * ptypes))
        self._lims = np.array(limits)
        self._ranges = self._lims[:, 1] - self._lims[:, 0]
        self._pts = np.random.random([self._n_pts, self._n_dims])

        # very much a hack
        # self._pts[:, -1] = np.around(self._pts[:, -1])
        # self._pts[:, -2] = np.around(self._pts[:, -2])
        self.count = np.zeros(self._n_pts)

    def __call__(self, x):
        xs = (x - self._lims[:, 0]) / self._ranges
        near = np.argpartition(np.max(np.abs(self._pts - xs), axis=1), self._k)
        k_nearest = near[:self._k]
        self.count[k_nearest] += 1

        # print("{:2.0f}    {:1.1f}    {:2.0f}".format(*self.spread))
        # if np.random.random() < 0.01:
        #     print(list(self.count))
        return k_nearest

    @property
    def spread(self):
        return self.count.min(), self.count.mean(), self.count.max()


# def kanerva_factory(sids: list, ptypes, num_active, action_in_state):
#     # dims = (5 + action_in_state) * len(sids)
#
#     # value limits of each dimension (min, max)
#     obs_lims = [(-1.7, 1.7),  # angular position (rads)
#                 (0, 7),  # speed (rads/s)
#                 (-1, 1)]  # load (unknown)
#                 # (0, 12),  # voltage (V)
#                 # (-10, 90)]  # temperature (C)
#     action_lims = [(-2, 2)]
#
#     dims = len(obs_lims) + action_in_state
#
#     # # one for each servo
#     # lims = sum([obs_lims for _ in sids], [])
#     # lims += sum([action_lims for _ in sids if action_in_state], [])
#     lims = obs_lims + (action_lims if action_in_state else [])
#
#     # create kanervacoder
#     kans = [KanervaCoder(dims,
#                          ptypes - 1,
#                          (num_active - 1) / (ptypes - 1),
#                          lims)
#             for _ in sids]
#
#     # add bias unit
#     def coder(x):
#         # return np.append(kan(x), [-1])
#
#         s = len(sids)
#
#         # grab pos, speed, load, and optionally action
#         xs = [x[i*5:i*5+3] + (x[i-s:i-s+1 if i-s+1 else None]
#                               if action_in_state
#                               else [])
#               for i in range(len(sids))]
#
#         return [np.append(kan(xi), [-1]) for kan, xi in zip(kans, xs)]
#
#     return coder, kans


# def kanerva_factory(sids: list, ptypes, num_active, action_in_state):
#     # JADEN
#     dims = 3 + action_in_state
#
#     # create kanervacoder
#     kans = [KanervaCoder(ptypes - 1,
#                          dims,
#                          'euclidian',
#                          (num_active - 1) / (ptypes - 1))
#             for _ in sids]
#
#     def coder(x):
#         # return np.append(kan(x), [-1])
#
#         # split x into per-servo features
#         n = len(x) // len(sids)
#
#         # grab pos, speed, load, and optionally action
#         xs = [x[i:i + 3] + (x[i + n - 1: i + n] if action_in_state else [])
#               for i in range(0, len(x), n)]
#
#         return [np.append(kan(xi), [-1]) for kan, xi in zip(kans, xs)]
#
#     return coder


def kanerva_factory(sids: list, action_in_state, *args, **kwargs):

    pos = np.asarray([i / 10 for i in range(-20, 20)])
    speed = np.asarray([i / 10 for i in range(0, 120)])
    load = np.array([-1, 1])

    chunk = len(pos) + len(speed)
    counts = np.zeros((4 * chunk, 2))

    def coder(x):
        # split x into per-servo features
        s = len(sids)

        # grab pos, speed, load, and optionally action
        xs = [x[i * 5:i * 5 + 3] + (x[i - s:i - s + 1 if i - s + 1 else None]
                                    if action_in_state
                                    else [])
              for i in range(len(sids))]

        chunks = []
        for xi in xs:
            p, s, l_, a = xi
            p = np.argmin(np.abs(pos - p))
            s = np.argmin(np.abs(speed - s))
            # p = pos.index(round(p * 10) / 10)
            # s = speed.index(round(s * 10) / 10)
            l_ = int(l_ == 1)
            a = int(a == 2)

            indices = [l_ * chunk + p,
                       l_ * chunk + p + pos.size + s,
                       load.size * chunk + a * chunk + p,
                       load.size * chunk + a * chunk + p + pos.size + s]

            counts[indices, int(bool(len(chunks)))] += 1

            indices.append(-1)

            chunks.append(indices)

        return chunks

    return coder, counts


async def data_from_file(main2gvf: mp.SimpleQueue,
                         gvf2plot: mp.SimpleQueue,
                         coder: KanervaCoder):
    data = np.load('offline_data.npy')

    for item in data:
        # print(item)
        item[-1] = coder(item[-2])
        main2gvf.put(item)

    time.sleep(0.1)
    while not gvf2plot.empty():
        time.sleep(0.1)


async def servo_loop(device: str,
                     sids: Sequence[int],
                     coder: KanervaCoder,
                     main2gvf: mp.SimpleQueue,
                     behaviour_policy: DiscretePolicy,
                     **kwargs):
    # objects to read and write from servos
    sr, sw = await serial_asyncio.open_serial_connection(url=device,
                                                         **kwargs)

    # set servo speeds to slowest possible
    for sid in sids:
        await send_msg(sr, sw, sid, [0x03, 0x20, 0x00, 0x01])

    # set initial action
    action = initial_action

    # some constants
    read_data = [0x02,  # read
                 0x24,  # starting from 0x24
                 0x08]  # a string of 8 bytes

    # read_all = [0x02,  # read
    #             0x00,  # starting from the beginning
    #             0x32]  # all the bytes

    store_data = []

    try:
        for _ in range(20000):
            # read data from servos
            byte_data = [await send_msg(sr, sw, sid, read_data) for sid in sids]

            # convert to human-readable data
            obs = sum([parse_data(bd) for bd in byte_data], []) + list(action)

            # get active tiles in kanerva coding
            active_pts = coder(obs)

            # get most recent weights from control GVFs
            pass

            # decide on an action
            action, action_prob = behaviour_policy(obs=obs, x=active_pts)

            # send action to servos
            instructions = [goal_instruction(a)
                            for a in action
                            if a is not None]
            for sid, instr in zip(sids, instructions):
                await send_msg(sr, sw, sid, instr)

            # send action and features to GVFs
            gvf_data = (action, action_prob, obs, active_pts)
            main2gvf.put(gvf_data)

            # record data for later
            store_data.append(gvf_data)

        np.save('offline_data.npy', store_data)

    except KeyboardInterrupt:
        pass
    finally:
        sr.read()
        await sw.drain()

        for sid in sids:
            write(sw, sid, [0x03, 0x18, 0x00])  # disable torque


def learning_loop(exit_flag: mp.Value,
                  gvfs: Sequence[Sequence[GTDLearner]],
                  main2gvf: mp.SimpleQueue,
                  gvf2plot: mp.SimpleQueue):
    action, action_prob, obs, x = None, None, None, None

    # get first state
    while exit_flag.value == 0 and obs is None:
        while exit_flag.value == 0 and main2gvf.empty():
            time.sleep(0.001)
        if exit_flag.value == 0:
            action, action_prob, obs, x = main2gvf.get()

    i = 1

    # main loop
    while exit_flag.value == 0:
        while exit_flag.value == 0 and main2gvf.empty():
            time.sleep(0.001)
        if exit_flag.value:
            break

        i += 1
        ude = False
        rupee = False
        if 5000 < i < 5100:
            ude = True
        if i == 7000:
            rupee = True

        # get data from servos
        actionp, action_probp, obsp, xp = main2gvf.get()

        # update weights
        for gs, xi, xpi in zip(gvfs, x, xp):
            for g in gs:
                g.update(action, action_prob, obs, obsp, xi, xpi, ude, rupee)

        # send data to plots
        gdata = [[g.data(xi, obs, action, xpi, obsp)
                  for g in gs]
                 for gs, xi, xpi in zip(gvfs, x, xp)]
        data = dict(ChainMap(*chain.from_iterable(gdata)))
        data['obs'] = obs
        gvf2plot.put(data)

        # go to next state
        obs = obsp
        x = xp
        action = actionp
        action_prob = action_probp

    print('Done learning!')


def plotting_loop(exit_flag: mp.Value,
                  gvf2plot: mp.SimpleQueue,
                  plots: Sequence[Plot]):

    while exit_flag.value == 0:
        while exit_flag.value == 0 and gvf2plot.empty():
            time.sleep(0.001)
        if exit_flag.value:
            break
        data = gvf2plot.get()

        for plot in plots:
            plot.update(data)

    for plot in plots:
        index = np.arange(len(plot.y[0]))
        np.savetxt(f"{plot.title}.csv",
                   np.column_stack(sum(((np.asarray(y),) for y in plot.y),
                                       (index,))),
                   delimiter=',')


def waver_action(obs: obsv,
                 predictions: Sequence[float]=None,
                 *args, **kwargs):
    del args, kwargs

    n = len(obs) // 5
    last_action = obs[-n:]
    action = []
    pavlov = [False] * n
    predictions = [0] * n if predictions is None else predictions

    # loop through each servo's 5-tuple
    for i in range(n):
        pos = obs[i * 5]
        if np.abs(predictions[i]) > 1.2:
            action.append(-2 * np.sign(pos))
            pavlov[i] = True
        else:
            if pos > 1.2:
                action.append(-2)
            elif pos < -1.2:
                action.append(2)
            else:
                action.append(last_action[i])

    return action, pavlov


def waver(actions: Sequence[actn],
          obs: obsv,
          *args, **kwargs) -> Sequence[float]:
    del args, kwargs

    action, _ = waver_action(obs)

    index = actions.index(tuple(action))
    probs = [0] * len(actions)
    probs[index] = 1

    return probs


def pavlov_waver(actions: Sequence[actn],
                 weights: Sequence[np.ndarray],
                 obs: obsv,
                 x: Dict[str, np.ndarray],
                 *args, **kwargs):
    del args, kwargs
    predictions = [w[xi].sum() for w, xi in zip(weights, x)]

    action, pavlov = waver_action(obs, predictions)

    index = actions.index(tuple(action))
    probs = [0] * len(actions)
    probs[index] = 1

    return probs, pavlov


def epsilon_policy(policy: Callable,
                   actions: Sequence[actn],
                   eps: float=0.1,
                   *args, **kwargs):

    # get action that was chosen by pavlov policy
    pavlov = [False for _ in range(len(actions[0]))]
    try:
        probs, pavlov = policy(actions=actions, *args, **kwargs)
    except ValueError:
        probs = policy(actions=actions, *args, **kwargs)
    action_index, *_ = [i for i in range(len(actions)) if probs[i]]
    action = actions[action_index]

    # get marginal probabilities for each action
    marginal_probs = [[((1 if a[i] == action[i] else 0)
                        if pavlov[i] else
                        ((1 - eps) if a[i] == action[i] else eps))
                       for i in range(len(actions[0]))]
                      for a in actions]

    return [np.prod(action_probs) for action_probs in marginal_probs]


def random_policy(actions: Sequence[actn], *args, **kwargs) -> Sequence[float]:
    del args, kwargs
    n = len(actions)
    return [1 / n for _ in actions]


def discounted_sum(gammas: Sequence[float],
                   x: Sequence[float]):
    sm = x[0]
    gam = 1

    for i in range(1, min(len(gammas), len(x)) - 1):
        gam *= gammas[i]
        sm += x[i] * gam

    return sm


def verifier(gvf_name: str,
             wait_time: int) -> Callable[[dat], Tuple[float, float]]:
    predictions = deque([], wait_time)
    cumulants = deque([], wait_time)
    gammas = deque([], wait_time)

    def f(data: dat) -> Tuple[float, float]:
        predictions.append(data[gvf_name]['pred'])
        cumulants.append(data[gvf_name]['cumul'])
        gammas.append(data[gvf_name]['gammap'])

        return predictions[0], discounted_sum(gammas, cumulants)

    return f


def gen_evals(alpha: float,
              n_features: int,
              alpha_h_hat: float=None,
              beta0_r: float=None,
              beta0_u: float=None,
              *args, **kwargs) -> Dict[str, Evaluator]:
    del args, kwargs
    alpha_h_hat = (10 * alpha
                   if alpha_h_hat is None
                   else alpha_h_hat)
    beta0_r = alpha / 50 if beta0_r is None else beta0_r
    rupee = RUPEE(n_features, beta0_r, alpha_h_hat)

    beta0_u = alpha * 10 if beta0_u is None else beta0_u
    ude = UDE(beta0_u)

    return {'rupee': rupee, 'ude': ude}


def hist_gamma(ind, default):
    last_act = None

    def gam(obs):
        nonlocal last_act
        if obs[ind] == last_act:
            last_act = default
        else:
            last_act = 0

        return last_act
    return gam


def name_gvf(kwargs):
    policy = kwargs['policy_name']
    servo = kwargs['servo']
    questn = kwargs['question']
    kwargs['name'] = f'{questn.capitalize()}-{policy}-{servo}'

    return kwargs


if __name__ == "__main__":
    # servo_loop variables
    servo_ids = [2, 3]
    serial_device = '/dev/tty.usbserial-AI03QEZV'
    m2g = mp.SimpleQueue()
    m2p = mp.SimpleQueue()
    g2p = mp.SimpleQueue()
    eflag = mp.Value('b', False)

    # behaviour policy
    actns = list(product([-2, 2], repeat=2))
    initial_action = (2, -2)
    random_turn = DiscretePolicy(actns,
                                 partial(epsilon_policy, waver, actns))
    turn = DiscretePolicy(actns, partial(waver, actns))
    random = DiscretePolicy(actns, partial(random_policy, actns))

    # gvfs
    num_features = 160 * 4 + 1  # 1025
    num_active_features = 5  # 33
    kanerva_coder, ks = kanerva_factory(servo_ids,
                                        ptypes=num_features,
                                        num_active=num_active_features,
                                        action_in_state=True)
    alpa = 0.1 / num_active_features
    policies = [
                {'target_policy': random,
                 'policy_name': 'random'},
                {'target_policy': turn,
                 'policy_name': 'wave'},
                {'target_policy': random_turn,
                 'policy_name': 'randomwave'}]

    # sum of near-in-time positions
    soon_pos = [{'alpha': alpa,
                 'gamma': lambda o: 0.75,
                 'lamda': 0.9,
                 'n_features': num_features,
                 'question': 'position'}]
    soon_cumuls = [{'cumulant': lambda x, o, a, xp, op: op[5],
                    'servo': 2},
                   {'cumulant': lambda x, o, a, xp, op: op[0],
                    'servo': 1}]
    params = product(soon_pos, soon_cumuls, policies)

    # time until switching direction
    time_switch = [{'alpha': alpa,
                    'lamda': 0.9,
                    'cumulant': lambda x, o, a, xp, op: 1,
                    'n_features': num_features,
                    'question': 'switchtime'}]
    time_gammas = [{'gamma': hist_gamma(-1, 1),
                    'servo': 2},
                   {'gamma': hist_gamma(-2, 1),
                    'servo': 1}]
    params = chain(params, product(time_switch, time_gammas, policies))

    # time until switching direction, else 0.8 gamma
    soft_switch = [{'alpha': alpa,
                    'lamda': 0.9,
                    'cumulant': lambda x, o, a, xp, op: 1,
                    'n_features': num_features,
                    'question': 'softswitch'}]
    soft_gammas = [{'gamma': hist_gamma(-1, 0.8),
                    'servo': 2},
                   {'gamma': hist_gamma(-2, 0.8),
                    'servo': 1}]
    params = chain(params, product(soft_switch, soft_gammas, policies))

    # nonsense required to make all the gvfs
    gvf_params = [name_gvf(dict(ChainMap(*p))) for p in params]
    gvf_list = [[GTDLearner(evaluators=gen_evals(**kwargs), **kwargs)
                 for kwargs in gvf_params
                 if f'{i + 1}' in kwargs['name']]
                for i in range(len(servo_ids))]

    # define learning loop
    learning_process = mp.Process(target=learning_loop,
                                  name='learning_loop',
                                  args=(eflag, gvf_list, m2g, g2p))

    # plotting
    vis = visdom.Visdom()
    vis.close(win=None)

    # get relevant gvf names and make plots for each horde in horde_names
    horde_names = 'position', 'softswitch', 'switchtime'
    horde_names = (horde_name.capitalize() for horde_name in horde_names)

    pos_plot = [
                Plot(parser=lambda data: (data['obs'][0],
                                          data['obs'][5]),
                     title='Position',
                     legend=['Servo 1', 'Servo 2'],
                     ylab='Angle'),
                # Plot(parser=lambda data: (data['obs'][2], data['obs'][7]),
                #      title='Load',
                #      legend=['Servo 1', 'Servo 2'],
                #      ylab='Load'),
                # Plot(parser=lambda data: [data[g.name]['rho']
                #                           for g in chain.from_iterable(gvf_list)
                #                           if 'Position-wave' in g.name],
                #      title='Rho',
                #      legend=['Servo 1', 'Servo 2'],
                #      ylab='Rho'),
                Plot(parser=verifier('Position-randomwave-1', 50),
                     title='Position-randomwave-1 verifier',
                     legend=['Predicted', 'Actual'],
                     ylab='Value'),
                Plot(parser=verifier('Position-randomwave-2', 50),
                     title='Position-randomwave-2 verifier',
                     legend=['Predicted', 'Actual'],
                     ylab='Value'),
                Plot(parser=lambda data: (data['obs'][4], data['obs'][9]),
                     title='Temp',
                     legend=['Servo 1', 'Servo 2'],
                     ylab='Temperature',
                     size=200),
                # Plot(parser=lambda data: [data[gname]['tderr']
                #                           for gname in ['Position-randomwave-1',
                #                                         'Position-randomwave-2']],
                #      title='TD errors',
                #      legend=['Position-randomwave-1', 'Position-randomwave-2'],
                #      ylab='TD Error',
                #      size=100),
                ]
    plts = pos_plot
    plts = sum((horde_plots(gvf_names=[gp['name']
                                       for gp in gvf_params
                                       if horde_name in gp['name']],
                            name=horde_name)
                for horde_name in horde_names),
               pos_plot)

    plotting_process = mp.Process(target=plotting_loop,
                                  name="plotting",
                                  args=(eflag, g2p, plts))

    pavlov_weights = [g.w
                      for g in chain.from_iterable(gvf_list)
                      if 'Position-wave' in g.name]
    pavlov_turner = partial(pavlov_waver, weights=pavlov_weights)
    behaviour = DiscretePolicy(actns,
                               partial(epsilon_policy, pavlov_turner, actns))
    # main coroutine setup
    loop = asyncio.get_event_loop()
    # coro = servo_loop(device=serial_device,
    #                   baudrate=1000000,
    #                   sids=servo_ids,
    #                   coder=kanerva_coder,
    #                   main2gvf=m2g,
    #                   behaviour_policy=behaviour
    #                   )
    coro = data_from_file(main2gvf=m2g, gvf2plot=g2p, coder=kanerva_coder)

    # start processes
    plotting_process.start()
    learning_process.start()

    # main loop
    try:
        loop.run_until_complete(coro)
        # np.savetxt("kanerva_counts.csv",
        #            np.column_stack((k.count for k in ks)),
        #            delimiter=',')
        # np.savetxt("bin_counts.csv",
        #            ks,
        #            delimiter=',')
    except KeyboardInterrupt:
        pass
    finally:
        eflag.value = 1
        plotting_process.join()
        learning_process.join()
        loop.close()
