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
from itertools import product, combinations, chain
from typing import (Callable, Iterable, List, NewType, Sequence, Tuple,
                    Dict, Union)

import numba as nb
import numpy as np
import serial_asyncio
from scipy.special import comb
import visdom


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


def binary_data(data):
    return [int(j) for d in data for j in bin(int(d))[2:].zfill(8)]


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
KanervaCoder = NewType('kan', Callable)


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
               delta_e: np.ndarray = None,
               x: np.ndarray = None,
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

    def update(self, delta: float = None, **kwargs):
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
#                 cumulant: Callable[[np.ndarray, obsv, actn, np.ndarray, obsv],
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
#       rho = self.target_policy.prob(w=w, x=x, obs=obs, a=action) / action_prob
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
#        self.eh = glrp * self.eh + self.alpha_h * (1 - glrp * self.eh[x].sum())
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
#        return {self.name: {**{k: v.value for k, v in self.evaluators.items()},
#                           **{'cumul': self.cumulant(x, obs, action, xp, obsp),
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
                 evaluators: Dict[str, Evaluator] = None,
                 alpha_h: float = None,
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
               ude: bool = False,
               rupee: bool = False):
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
                            'gamma': self.gamma(obs),
                            'gammap': self.gamma(obsp)}}


class Horde:
    def __init__(self,
                 alpha: float,
                 gamma: Callable[[obsv], float],
                 cumulants: Callable[[np.ndarray, obsv, actn, np.ndarray, obsv],
                                     List[float]],
                 lamda: float,
                 n_features: int,
                 names: List[str],
                 evaluators: Dict[str, Evaluator] = None,
                 *args, **kwargs):
        del args, kwargs

        self.α = alpha
        self.γ = gamma
        self.λ = lamda
        self.cumulants = cumulants
        self.names = names

        self.w = np.zeros((len(names), n_features))
        self.e = np.zeros((len(names), n_features))
        self.h = np.zeros((len(names), n_features))
        self.la = initial_action
        self.v_old = np.zeros(len(names))

        self.evaluators = {} if evaluators is None else evaluators
        self.δ = None

    def update(self,
               action: actn,
               action_prob: float,
               obs: obsv,
               obsp: obsv,
               x: np.ndarray,
               xp: np.ndarray,
               *args, **kwargs):
        del action_prob, args, kwargs
        # for readability
        α, w, e, h, λ = self.α, self.w, self.e, self.h, self.λ

        # precalculations
        γ = self.γ(obs)
        γp = self.γ(obsp)
        reward = self.cumulants(x, obs, action, xp, obsp)
        # x = x[-3:]
        # xp = xp[-3:]
        v = sum_axis1(w, x)
        vp = sum_axis1(w, xp)
        self.δ = reward + γp * vp - v

        # true online TD
        e_x = sum_axis1(e, x)
        e *= γ * λ
        update_axis1(e, x, 1 - α * γ * λ * e_x)
        w += α * np.einsum('i,ij-> ij', self.δ + v - self.v_old, e)
        update_axis1(w, x, -α * (v - self.v_old))
        self.v_old = vp

        # evaluators
        ev_kwargs = {'x': x,
                     'delta': self.δ}
        for k, ev in self.evaluators.items():
            ev.update(**ev_kwargs)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return sum_axis1(self.w, x)

    def data(self,
             x: np.ndarray,
             obs: obsv,
             action: actn,
             xp: np.ndarray,
             obsp: obsv) -> Dict[str, Dict[str, float]]:
        pred = self.predict(x)
        cumuls = self.cumulants(x, obs, action, xp, obsp)
        return {name: {**{k: v.value for k, v in self.evaluators.items()},
                       'cumul': cumuls[i],
                       'pred': pred[i],
                       'tderr': self.δ[i],
                       'gamma': self.γ(obs),
                       'gammap': self.γ(obsp)}
                for i, name in enumerate(self.names)}


@nb.jit(nb.float64[:](nb.float64[:], nb.int64[:]), nogil=True)
def sum_axis1(a, x):
    ssum = np.zeros(a.shape[0], dtype=np.float64)
    for i in range(a.shape[0]):
        for j in range(x.size):
            ssum[i] += a[i, x[j]]
    return ssum


@nb.jit(nb.float64[:](nb.float64[:], nb.int64[:], nb.float64), nogil=True)
def update_axis1(a, x, b):
    for i in range(a.shape[0]):
        for j in range(x.size):
            a[i, x[j]] += b[i]


class Plot:
    def __init__(self,
                 parser: Callable[[Sequence[Sequence[float]]], Iterable[float]],
                 title: str,
                 legend: List[str]=None,
                 *args, **kwargs):
        self.parser = parser
        self.title = title
        self.y = [[] for _ in range(len(legend))]

    def update(self, *args, **kwargs):
        pass


class LinePlot(Plot):
    def __init__(self,
                 parser: Callable[[Sequence[Sequence[float]]], Iterable[float]],
                 title: str,
                 legend: List[str],
                 size: int = 50,
                 *args, **kwargs):
        del args
        super().__init__(parser, title, legend)

        layoutopts = {'plotly':
                      {'legend': legend}}

        self.opts = dict(ChainMap(*[kwargs,
                                    dict(title=self.title,
                                         colormap='Inferno',
                                         layoutopts=layoutopts,
                                         ylab=kwargs.get('ylab', ''),
                                         xlab=kwargs.get('xlab', 'Time-step'),
                                         )
                                    ]
                                  )
                         )

        self.i = 0
        self.y_windows = [deque(size * [0], size)
                          for _ in range(len(legend))]
        self.predicted_window = deque(size * [0], size)
        self.x_window = deque([0 for _ in range(size)], size)

        self.moving_window = vis.line(
            X=np.asarray([0, 0]),
            Y=np.column_stack([[0, 0]] * len(legend)),
            opts=self.opts,
        )

    def update(self, data: dat):
        self.i += 1

        for y_list in self.y_windows, self.y:
            for y, a in zip(y_list, data):
                y.append(a)
        self.x_window.append(self.i)

        # Update windowed graph
        # vis.updateTrace([self.i for _ in args], args, self.moving_window)
        vis.line(
            X=np.asarray(self.x_window),
            Y=np.column_stack(self.y_windows),
            win=self.moving_window,
            update='replace',
            opts=self.opts)
        # vis.line(
        #     X=np.asarray([self.i for _ in data]),
        #     Y=np.asarray(data),
        #     win=self.moving_window,
        #     update='append',
        #     opts=self.opts)


class HeatMap(Plot):
    def __init__(self,
                 parser: Callable[[Sequence[Sequence[float]]], Iterable[float]],
                 title: str,
                 legend: List[str],
                 num_rows: int=4,
                 num_cols: int=100,
                 zrange: Tuple[float, float]=(0, 1),
                 *args, **kwargs):
        del args
        super().__init__(parser, title, legend)
        self.num_rows = num_rows
        self.num_cols = num_cols

        layoutopts = {'plotly':
                      {'xaxis': dict(autorange=True,
                                     showgrid=False,
                                     zeroline=False,
                                     showline=False,
                                     autotick=True,
                                     ticks='',
                                     showticklabels=False)
                       }
                      }

        self.opts = dict(ChainMap(*[kwargs,
                                    dict(title=self.title,
                                         xmin=zrange[0],
                                         xmax=zrange[1],
                                         colormap='Inferno',
                                         layoutopts=layoutopts)]))

        self.moving_window = vis.heatmap(
            X=np.zeros((num_rows, num_cols)),
            opts=self.opts,
        )

    def update(self, data: dat):

        for y, a in zip(self.y, [data]):
            y.append(a)

        matrix = np.asarray(data).reshape(self.num_cols, self.num_rows).T

        # Update windowed graph
        vis.heatmap(X=matrix, win=self.moving_window, opts=self.opts)


def horde_plots(gvf_names: List[str], name: str) -> List[Plot]:

    def get_phi(data):
        phi = np.zeros(num_features)
        phi[data['x'][0]] = 1
        return phi[1:401]

    return [HeatMap(parser=get_phi,
                    title=f'Bit Values',
                    legend=gvf_names,
                    num_rows=1,
                    num_cols=num_bits,
                    height=30+30+50,
                    width=1420,
                    marginleft=50,
                    marginright=40,
                    margintop=40,
                    marginbottom=40,
                    showlegend=False),

            HeatMap(parser=lambda data: [data[gname]['pred']
                                         for gname in gvf_names],
                    title=f'{name.capitalize()} Prediction',
                    legend=gvf_names,
                    num_rows=num_gammas,
                    num_cols=num_bits,
                    zrange=(0, 20),
                    height=30+30+160,
                    width=1500,
                    marginleft=40,
                    marginright=40,
                    margintop=40,
                    marginbottom=40),
            HeatMap(parser=lambda data: [data[gname]['tderr']
                                         for gname in gvf_names],
                    title=f'{name.capitalize()} TD error',
                    legend=gvf_names,
                    num_rows=num_gammas,
                    num_cols=num_bits,
                    zrange=(-2, 2),
                    height=30+30+160,
                    width=1500,
                    marginleft=40,
                    marginright=40,
                    margintop=40,
                    marginbottom=40),
            ]


def kanerva_factory(sids: list, *args, **kwargs) -> KanervaCoder:
    del args, kwargs

    pos = np.asarray([i / 10 for i in range(-20, 20)])
    speed = np.asarray([i / 10 for i in range(0, 120)])
    load = np.array([-1, 1])

    chunk_size = len(pos) + len(speed)
    counts = np.zeros(load.size * chunk_size)

    def coder(x2, x1):
        # grab pos, speed, load
        xs = [x2[i * 5 + 1:i * 5 + 4] for i in range(len(sids))]

        chunks = []
        for xi in xs:
            p, s, l_, *_ = xi
            p = np.argmin(np.abs(pos - p))
            s = np.argmin(np.abs(speed - s))
            # p = pos.index(round(p * 10) / 10)
            # s = speed.index(round(s * 10) / 10)
            l_ = int(l_ == 1)

            indices = [l_ * chunk_size + p,
                       l_ * chunk_size + pos.size + s]

            counts[indices] += 1
            indices.append(-1)

            chunks.append(np.asarray(indices))

        return [np.concatenate((x1[0], chunks[0] + 400))]

    return coder, counts


def add_combinations(x1: List[np.ndarray], ks: List[int], *args, **kwargs):
    del args, kwargs
    # assert ks == [2]

    changing = [241, 242, 243, 244, 245, 246, 247, 255, 256, 289, 290, 291,
                292, 293, 294, 295, 296, 303, 304, 305, 306, 307, 308, 309,
                310, 318, 320, 321, 322, 323, 324, 325, 334, 335, 336, 342,
                343, 344, 348, 349, 350, 351, 352, 376]

    combos = chain.from_iterable(combinations(changing, k) for k in ks)
    active_combos = [int(num_bits + i)
                     for i, cs in enumerate(combos)
                     if all(c in x1[0] for c in cs)]
    return [np.concatenate((x1[0], active_combos)).astype(int)]


async def data_from_file(main2gvf: mp.SimpleQueue,
                         coder: KanervaCoder):
    data = np.load('offline_data.npy')
    for i, item in enumerate(data):
        # if i > 500:
        #     break
        item[-1] = coder(x1=item[-1], x2=item[-2])
        main2gvf.put(item)


def binary_pts(byte_data, obs=None):
    del obs
    phi = np.asarray(sum([binary_data(bd) for bd in byte_data], [1]))
    return [np.where(phi)[0]]


async def servo_loop(device: str,
                     sids: Sequence[int],
                     main2gvf: mp.SimpleQueue,
                     behaviour_policy: DiscretePolicy,
                     coder: KanervaCoder,
                     **kwargs):
    # objects to read and write from servos
    sr, sw = await serial_asyncio.open_serial_connection(url=device,
                                                         **kwargs)

    # set servo speeds to slowest possible
    for sid in sids:
        # await send_msg(sr, sw, sid, [])
        await send_msg(sr, sw, sid, [0x03, 0x20, 0x00, 0x01])

    # set initial action
    action = initial_action

    # some constants
    # read_data = [0x02,  # read
    #              0x24,  # starting from 0x24
    #              0x08]  # a string of 8 bytes

    read_all = [0x02,  # read
                0x00,  # starting from the beginning
                0x32]  # all the bytes

    store_data = []

    try:
        for _ in range(20000):
            # read data from servos
            byte_data = [await send_msg(sr, sw, sid, read_all) for sid in sids]

            # convert to human-readable data
            obs = sum([parse_data(bd) for bd in byte_data], list(action))

            # make feature vector
            active_pts = coder(obs=obs, byte_data=byte_data)

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
            if locks:
                print('main gm a 1 a')
                gmlock.acquire()
                print('main gm a 1 b')
            main2gvf.put(gvf_data)
            if locks:
                print('main gm r 1 a')
                gmlock.release()
                print('main gm r 1 b')

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
                  gvf2plot: mp.SimpleQueue,
                  parsrs: List[Callable]):
    action, action_prob, obs, x = None, None, None, None

    # get first state
    while exit_flag.value == 0 and obs is None:
        while exit_flag.value == 0 and main2gvf.empty():
            time.sleep(0.01)
        if exit_flag.value == 0:
            if locks:
                print('gvf gm a 1 a')
                gmlock.acquire()
                print('gvf gm a 1 b')
            action, action_prob, obs, x = main2gvf.get()
            if locks:
                print('gvf gm r 1 a')
                gmlock.release()
                print('gvf gm r 1 b')

    # main loop
    # tt = 0
    # ts = []
    while exit_flag.value == 0:
        # ts.append(time.time() - tt) if tt > 0 else None
        # print(np.mean(ts))
        # tt = time.time()
        if locks:
            print('gvf gm a 2 a')
            gmlock.acquire()
            print('gvf gm a 2 b')
        while exit_flag.value == 0 and main2gvf.empty():
            if locks:
                print('gvf gm r 2 a')
                gmlock.release()
                print('gvf gm r 2 b')
            time.sleep(0.01)
            if locks:
                print('gvf gm a 3 a')
                gmlock.acquire()
                print('gvf gm a 3 b')
        if locks:
            print('gvf gm r 3 a')
            gmlock.release()
            print('gvf gm r 3 b')
        if exit_flag.value:
            break

        # get data from servos
        if locks:
            print('gvf gm a 4 a')
            gmlock.acquire()
            print('gvf gm a 4 b')
        actionp, action_probp, obsp, xp = main2gvf.get()
        if locks:
            print('gvf gm r 4 a')
            gmlock.release()
            print('gvf gm r 4 b')
        # update weights
        for gs, xi, xpi in zip(gvfs, x, xp):
            for g in gs:
                g.update(action, action_prob, obs, obsp, xi, xpi)

        # send data to plots
        gdata = [g.data(xi, obs, action, xpi, obsp)
                 for gs, xi, xpi in zip(gvfs, x, xp)
                 for g in gs]

        data = dict(ChainMap(*gdata))
        data['obs'] = obs
        data['x'] = x
        data = [parse(data) for parse in parsrs]
        if locks:
            print('gvf gp a 1 a')
            gplock.acquire()
            print('gvf gp a 1 b')
        # data = np.copy(data)
        gvf2plot.put(data)
        if locks:
            print('gvf gp r 1 a')
            gplock.release()
            print('gvf gp r 1 b')

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
        if locks:
            print('plot gp a 1 a')
            gplock.acquire()
            print('plot gp a 1 b')
        while exit_flag.value == 0 and gvf2plot.empty():
            if locks:
                print('plot gp r 1 a')
                gplock.release()
                print('plot gp r 1 b')
            time.sleep(0.001)
            if locks:
                print('plot gp a 2 a')
                gplock.acquire()
                print('plot gp a 2 b')

        if locks:
            print('plot gp r 2 a')
            gplock.release()
            print('plot gp r 2 b')
        if exit_flag.value:
            break

        if locks:
            print('plot gp a 3 a')
            gplock.acquire()
            print('plot gp a 3 b')
        d = gvf2plot.get()
        if locks:
            print('plot gp r 3 a')
            gplock.release()
            print('plot gp r 3 b')

        for plot, data in zip(plots, d):
            plot.update(data)

    for plot in plots:
        try:
            index = np.arange(len(plot.y[0]))
            np.savetxt(f"{plot.title}.csv",
                       np.column_stack(sum(((np.asarray(y),) for y in plot.y),
                                           (index,))),
                       delimiter=',')
        except ValueError:
            continue


def waver_action(obs: obsv,
                 predictions: Sequence[float] = None,
                 *args, **kwargs):
    del args, kwargs

    a = 1.5
    n = len(obs) // 5
    last_action = obs[:-n]
    action = []
    pavlov = [False] * n
    predictions = [0] * n if predictions is None else predictions

    # loop through each servo's 5-tuple
    for i in range(n):
        pos = obs[i * 5 + n]
        if np.abs(predictions[i]) > 1.2:
            action.append(-a * np.sign(pos))
            pavlov[i] = True
        else:
            if pos > 1.2:
                action.append(-a)
            elif pos < -1.2:
                action.append(a)
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
                   eps: float = 0.1,
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


def horde_verifier(verifiers: List[List[Callable[[dat], Tuple[float, float]]]]):

    def f(d: dat) -> List[float]:
        x = np.asarray([[v(d) for v in vs] for vs in verifiers])
        return [np.sqrt(np.power(x[i, :, 1] - x[i, :, 0], 2).sum() / x.shape[1])
                for i in range(x.shape[0])]

    return f


def gen_evals(alpha: float,
              n_features: int,
              alpha_h_hat: float = None,
              beta0_r: float = None,
              beta0_u: float = None,
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
    gamma = kwargs['gamma'](0)
    kwargs['name'] = f'{questn.capitalize()}-{gamma}-{policy}-{servo}'

    return kwargs


if __name__ == "__main__":
    # servo_loop variables
    servo_ids = [2]
    serial_device = '/dev/tty.usbserial-AI03QEZV'
    m2g = mp.SimpleQueue()
    m2p = mp.SimpleQueue()
    g2p = mp.SimpleQueue()
    eflag = mp.Value('b', False)
    collect_data = False
    gmlock = mp.Lock()
    gplock = mp.Lock()
    locks = False

    # behaviour policy
    actns = list(product([-1.5, None, 1.5], repeat=1))
    initial_action = (1.5,)

    turn = DiscretePolicy(actns, partial(waver, actns))

    # gvfs
    num_bits = 50 * 8
    num_gammas = 5
    combin = [2, 3]
    num_features = int(num_bits + sum(comb(44, ki) for ki in combin) + 1)
    alpa = 0.1 / 5000
    lam = 0

    policies = [{'target_policy': turn,
                 'policy_name': 'wave'}]

    # make gvf questions
    on_policy = [{'alpha': alpa,
                  'beta': 0,
                  'cumulant': lambda x, o, a, xp, op, i=i: int(i+1 in xp),
                  'lamda': lam,
                  'n_features': num_features,
                  'servo': 1,
                  'question': f'Bit{i}'} for i in range(num_bits)]
    on_policy_gammas = [{'gamma': lambda o, n=n: 1 - np.power(2, -float(n))}
                        for n in range(num_gammas)]
    params = product(on_policy, on_policy_gammas, policies)

    # nonsense required to make all the gvfs
    gvf_params = [name_gvf(dict(ChainMap(*p))) for p in params]
    gvf_list = [[GTDLearner(evaluators=gen_evals(**kwargs), **kwargs)
                 for kwargs in gvf_params
                 if f'{i + 1}' in kwargs['name']]
                for i in range(len(servo_ids))]

    horde_params = {'alpha': alpa,
                    'lamda': lam,
                    'n_features': num_features,
                    }
    gvf_gammas = [lambda o, n=n: 1 - np.power(2, -float(n))
                  for n in range(num_gammas)]

    gnames = [[g.name
               for gs in gvf_list
               for g in gs
               if g.gamma(0) == gam(0)]
              for gam in gvf_gammas]
    gvf_cumulants = [lambda x, o, a, xp, op: [int(i+1 in xp)
                                              for i in range(num_bits)]
                     for _ in gvf_gammas]
    horde = [[Horde(gamma=g, cumulants=c, names=n, **horde_params)
              for g, c, n in zip(gvf_gammas, gvf_cumulants, gnames)]]

    # plotting
    vis = visdom.Visdom()
    vis.close(win=None)

    # plts = horde_plots([g.name for gs in gvf_list for g in gs], 'Horde')
    plts = [LinePlot(parser=horde_verifier([[verifier(g.name,
                                                       int(1/(1 - g.gamma(0))))
                                             for gs in gvf_list
                                             for g in gs
                                             if g.gamma(0) == gam(0)]
                                            for gam in gvf_gammas],),
                      title='Horde RMSE',
                      legend=[str(g(0)) for g in gvf_gammas],
                      ylab='',
                      showlegend=True,
                      size=10,
                      height=400,
                      width=700,
                      marginleft=50,
                      marginright=40,
                      margintop=40,
                      marginbottom=40
                      )
             ]

    parsers = [plot.parser for plot in plts]

    codr = partial(add_combinations, ks=combin)
    # codr = kanerva_factory(servo_ids)[0]

    # main coroutine setup
    loop = asyncio.get_event_loop()
    if collect_data:
        coro = servo_loop(device=serial_device,
                          baudrate=1000000,
                          sids=servo_ids,
                          main2gvf=m2g,
                          behaviour_policy=turn,
                          coder=kanerva_factory(servo_ids),
                          )
    else:
        coro = data_from_file(main2gvf=m2g,
                              coder=codr)

    # start processes
    learning_process = None
    plotting_process = None
    if not collect_data or collect_data:
        learning_process = mp.Process(target=learning_loop,
                                      name='learning_loop',
                                      args=(eflag, horde, m2g, g2p, parsers))

        plotting_process = mp.Process(target=plotting_loop,
                                      name="plotting",
                                      args=(eflag, g2p, plts))
        # import threading
        # learning_process = threading.Thread(target=learning_loop,
        #                               name='learning_loop',
    #                               args=(eflag, gvf_list, m2g, g2p, parsers))
        #
        # plotting_process = threading.Thread(target=plotting_loop,
        #                               name="plotting",
        #                               args=(eflag, g2p, plts))
        learning_process.start()
        plotting_process.start()
    # main loop
    try:
        t = time.time()
        loop.run_until_complete(coro)
        # np.savetxt("kanerva_counts.csv",
        #            np.column_stack((k.count for k in ks)),
        #            delimiter=',')
        # np.savetxt("bin_counts.csv",
        #            ks,
        #            delimiter=',')

        while not m2g.empty() or not g2p.empty():
            time.sleep(3)
        print(f'{time.time() - t}')
    except KeyboardInterrupt:
        pass
    finally:
        eflag.value = 1
        if not collect_data:
            learning_process.join()
            plotting_process.join()
        loop.close()
