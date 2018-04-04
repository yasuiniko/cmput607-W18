"""Plotting code is modified from code by Cam Linke.
Everything else by Niko Yasui.

Use Python 3.5 or higher.
"""
import asyncio
from collections import deque, ChainMap
import math
import multiprocessing as mp
import time
from itertools import product, chain
from typing import (Callable, Iterable, List, NewType, Sequence, Tuple,
                    Dict, Union)

import numpy as np
from scipy.stats import norm
import serial_asyncio
import visdom


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
            0x1e,  # to goal posit
            lo, hi]  # desired posit


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


class ContinuousPolicy:
    def __init__(self,
                 sampler: Callable[..., actn],
                 actn_prob: Callable[..., float],
                 actn_range: Sequence[float]):
        self.sampler = sampler
        self.prob = actn_prob
        self.low = actn_range[0]
        self.high = actn_range[1]

    def __call__(self, *args, **kwargs) -> (actn, float):
        raw_action = self.sampler(*args, **kwargs)
        action = max(min(raw_action, self.high), self.low)

        return action, self.prob(a=action, *args, **kwargs)


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


class Normalpdf:
    def __init__(self, x):
        self.μ = 0
        self.σ = 1
        self.x = x

    def update(self, μ: float, σ: float, **kwargs):
        del kwargs
        self.μ = μ
        self.σ = σ

    @property
    def value(self) -> List[float]:
        return [norm.pdf(xi, self.μ, self.σ) for xi in self.x]


class Variable:
    def __init__(self, var):
        self.var = var
        self.x = None

    def update(self, **kwargs):
        self.x = kwargs[self.var]

    @property
    def value(self) -> List[float]:
        return self.x


Evaluator = Union[RUPEE, UDE]


class Learner:
    def __init__(self,
                 alpha: float,
                 gamma: Callable[[obsv], float],
                 cumulant: Callable[[List[np.ndarray], obsv, actn,
                                     List[np.ndarray], obsv],
                                    float],
                 lamda: float,
                 actions: List[actn],
                 name: str,
                 sid: int,
                 evaluators: Dict[str, Evaluator] = None,
                 *args, **kwargs):
        del args, kwargs
        self.α = alpha * (1 - lamda)
        self.γ = gamma
        self.λ = lamda
        self.cumulant = cumulant
        self.name = name
        self.actions = actions

        self.sid = sid
        self.evaluators = {} if evaluators is None else evaluators

        self.w = None
        self.δ = None
        self.n = None

    def data(self,
             x: List[np.ndarray],
             obs: obsv,
             action: actn,
             xp: List[np.ndarray],
             obsp: obsv) -> Dict[str, Dict[str, float]]:
        pass

    def update(self,
               x: List[np.ndarray],
               obs: obsv,
               action: actn,
               action_prob: float,
               xp: List[np.ndarray],
               obsp: obsv,
               actionp: actn,
               action_probp: float,
               *args, **kwargs):
        pass

    def predict(self, x: np.ndarray, a: actn) -> float:
        pass


class ControlLearner(Learner):
    def policy(self, *args, **kwargs):
        pass

    def data(self,
             x: List[np.ndarray],
             obs: obsv,
             action: actn,
             xp: List[np.ndarray],
             obsp: obsv) -> Dict[str, Dict[str, float]]:
        return {self.name: {**{k: v.value for k, v in self.evaluators.items()},
                            **{'cumul': self.cumulant(x,
                                                      obs,
                                                      action,
                                                      xp,
                                                      obsp),
                               'pred': self.predict(x[self.sid],
                                                    action)},
                            'tderr': self.δ,
                            'gammap': self.γ(obsp),
                            'action': action}}


class SARSA(ControlLearner):
    def __init__(self,
                 n_features: int,
                 actions: List[actn],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.w = np.zeros(n_features * len(actions))
        self.e = np.zeros(n_features * len(actions))
        self.q_old = 0
        self.actions = actions
        self.n = n_features

        self.policy = DiscretePolicy(actions, self.opt)

    def update(self,
               x: List[np.ndarray],
               obs: obsv,
               action: actn,
               action_prob: float,
               xp: List[np.ndarray],
               obsp: obsv,
               actionp: actn,
               action_probp: float,
               *args, **kwargs):
        del args, kwargs, action_prob, action_probp
        # for readability
        alpha, w, e, lamda = self.α, self.w, self.e, self.λ

        # shift to proper sub-vector
        a_ind = self.actions.index(action)
        ap_ind = self.actions.index(actionp)
        xs, xps = x[self.sid], xp[self.sid]
        xa = xs + self.n * a_ind
        xa[-1] = -1
        xap = xps + self.n * ap_ind
        xap[-1] = -1

        # precalculations
        gamma = self.γ(obs)
        gammap = self.γ(obsp)
        reward = self.cumulant(x, obs, self.actions[a_ind], xp, obsp)
        q = w[xa].sum()
        qp = w[xap].sum()
        self.δ = reward + gammap * qp - q

        # True online SARSA
        ex = e[xa].sum()
        e *= gamma * lamda
        e[xa] += 1 - alpha * gamma * lamda * ex

        w += alpha * (self.δ + q - self.q_old) * e
        w[xa] -= alpha * (q - self.q_old)

        self.q_old = qp

        # evaluators
        ev_kwargs = {'x': x}
        for k, ev in self.evaluators.items():
            ev.update(**ev_kwargs)

    def predict(self, xs: np.ndarray, a: actn) -> float:
        xa = xs + self.n * self.actions.index(a)
        xa[-1] = -1
        return self.w[xa].sum()

    def opt(self, x, *args, **kwargs):
        del args, kwargs
        values = np.asarray([self.predict(x[self.sid], a)
                             for a in self.actions])
        maximal = values.max() == values
        return maximal / maximal.sum()


class DiscreteAC(ControlLearner):
    def __init__(self,
                 n_features: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.w = np.zeros(n_features)
        self.e_w = np.zeros(n_features)
        self.θ = np.zeros(n_features * len(self.actions))
        self.e_θ = np.zeros(n_features * len(self.actions))
        self.v_old = 0
        self.n = n_features

        self.policy = DiscretePolicy(self.actions, self.softmax)

    def softmax(self, x, *args, **kwargs):
        del args, kwargs
        x = x[self.sid]
        values = np.asarray([self.pick_action(x, a) for a in self.actions])
        e_v = np.exp(values - np.max(values))
        return e_v / e_v.sum()

    def action_shift(self, xs: np.ndarray, a_ind: int) -> np.ndarray:
        xa = xs + self.n * a_ind
        xa[-1] = -1
        return xa

    def pick_action(self, xs: np.ndarray, a: actn) -> float:
        xa = self.action_shift(xs, self.actions.index(a))
        return self.θ[xa].sum()

    def predict(self, xs: np.ndarray, a: actn) -> float:
        return self.w[xs].sum()

    def update(self,
               x: List[np.ndarray],
               obs: obsv,
               action: actn,
               action_prob: float,
               xp: List[np.ndarray],
               obsp: obsv,
               actionp: actn,
               action_probp: float,
               *args, **kwargs):
        del args, kwargs, action_prob, action_probp
        # for readability
        α, w, e_w, λ = self.α, self.w, self.e_w, self.λ
        θ, e_θ = self.θ, self.e_θ

        # shift to proper sub-vector
        a_ind = self.actions.index(action)
        xs, xps = x[self.sid], xp[self.sid]

        # precalculations
        γ = self.γ(obs)
        γp = self.γ(obsp)
        reward = self.cumulant(x, obs, self.actions[a_ind], xp, obsp)
        v = w[xs].sum()
        vp = w[xps].sum()
        self.δ = reward + γp * vp - v

        # critic update with true online TD
        e_wx = e_w[xs].sum()
        e_w *= γ * λ
        e_w[xs] += 1 - α * γ * λ * e_wx
        w += α * (self.δ + v - self.v_old) * e_w
        w[xs] -= α * (v - self.v_old)
        self.v_old = vp

        # actor update
        e_θ *= γ * λ
        e_θ += self.partial(a_ind, x)
        θ += α * self.δ * e_θ

        # evaluators
        ev_kwargs = {'x': x, 'action_probs': self.softmax(x)}
        for k, ev in self.evaluators.items():
            ev.update(**ev_kwargs)

    def partial(self, a_ind: int, x: List[np.ndarray]) -> np.ndarray:
        xs = x[self.sid]
        action_probs = self.softmax(x)

        # make sparse x(s, a)
        xa_sparse = self.action_shift(xs, a_ind)
        xa = np.zeros_like(self.θ)
        xa[xa_sparse] = 1

        # make x(s, b) for all b
        xbs_sparse = [self.action_shift(xs, b)
                      for b in range(len(self.actions))]
        xbs = np.zeros((len(self.actions), self.θ.size))
        for b in range(len(self.actions)):
            xbs[b, xbs_sparse[b]] = 1

        return xa - sum(xbi * bpi for xbi, bpi in zip(xbs, action_probs))


class ContinuousAC(ControlLearner):
    def __init__(self,
                 n_features: int,
                 actn_range: Sequence[actn]=((-1.5,), (1.5,)),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = np.zeros(n_features)
        self.e_w = np.zeros(n_features)
        self.θ = np.zeros((2, n_features))
        self.e_θ = np.zeros_like(self.θ)

        self.v_old = 0
        self.n = n_features

        self.policy = ContinuousPolicy(self.sample, self.aprob, actn_range)

    def update(self,
               x: List[np.ndarray],
               obs: obsv,
               action: actn,
               action_prob: float,
               xp: List[np.ndarray],
               obsp: obsv,
               actionp: actn,
               action_probp: float,
               *args, **kwargs):
        del args, kwargs, action_prob, action_probp
        # for readability
        α, w, e_w, λ = self.α, self.w, self.e_w, self.λ
        θ, e_θ = self.θ, self.e_θ

        # shift to proper sub-vector
        xs, xps = x[self.sid], xp[self.sid]

        # precalculations
        γ = self.γ(obs)
        γp = self.γ(obsp)
        reward = self.cumulant(x, obs, action, xp, obsp)
        v = w[xs].sum()
        vp = w[xps].sum()
        self.δ = reward + γp * vp - v

        # critic update with true online TD
        e_wx = e_w[xs].sum()
        e_w *= γ * λ
        e_w[xs] += 1 - α * γ * λ * e_wx
        w += α * (self.δ + v - self.v_old) * e_w
        w[xs] -= α * (v - self.v_old)
        self.v_old = vp

        # actor update
        e_θ *= γ * λ
        grad, μ, σ = self.partial(action, x)
        e_θ += grad
        θ += 0.1 * α * self.δ * e_θ

        # evaluators
        ev_kwargs = {'x': x, 'μ': μ, 'σ': σ}
        for k, ev in self.evaluators.items():
            ev.update(**ev_kwargs)

    def get_params(self, xs: np.ndarray):
        μ, σ = self.θ[0][xs].sum(), np.exp(self.θ[1][xs].sum())
        assert σ > 0
        return μ, σ

    def sample(self, x: List[np.ndarray], *args, **kwargs):
        del args, kwargs
        μ, σ = self.get_params(x[self.sid])
        return np.random.normal(μ, σ),

    def aprob(self, x: List[np.ndarray], a: actn, *args, **kwargs):
        del args, kwargs
        μ, σ = self.get_params(x[self.sid])
        return norm.pdf(a, μ, σ)

    def partial(self, action: actn, x: List[np.ndarray]) -> Tuple:
        # set up
        xs_pts = x[self.sid]
        x_μ = np.zeros(self.n)
        x_μ[xs_pts] = 1
        x_σ = np.copy(x_μ)

        # get parameters
        μ, σ = self.get_params(x[self.sid])

        # find derivatives
        σ2 = np.power(σ, 2)
        dμ = x_μ * (action - μ) / σ2
        # print(self.δ * (np.power(action - μ, 2) / σ2 - 1))
        dσ = x_σ * (np.power(action - μ, 2) / σ2 - 1)

        return np.vstack((dμ, dσ)), μ, σ

    def predict(self, xs: np.ndarray, *args, **kwargs) -> float:
        del args, kwargs
        return self.w[xs].sum()


class Plot:
    def __init__(self,
                 parser: Callable[[Sequence[Sequence[float]]], Iterable[float]],
                 title: str,
                 legend: List[str],
                 ylab: str,
                 xlab: str='Time-step',
                 sizes: Sequence[int]=(50,)):
        self.parser = parser
        self.title = title

        self.y_windows = [[deque(size * [0], size)
                           for _ in range(len(legend))]
                          for size in sizes]
        self.x_windows = [deque(size * [0], size) for size in sizes]

        self.moving_windows = [vis.line(
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
            for _ in sizes]

        self.i = 0
        self.y = [[[] for _ in range(len(legend))] for _ in sizes]

    def update(self, data: dat):
        try:
            self.update_plot(*self.parser(data))
        except:
            print(data)
            print(self.parser(data))
            raise

    def update_plot(self, *args: float):
        self.i += 1

        for y_list in self.y_windows, self.y:
            for ys in y_list:
                for y, a in zip(ys, args):
                    y.append(a)

        for x in self.x_windows:
            x.append(self.i)

        # Update windowed graph
        for x, y, m in zip(self.x_windows, self.y_windows, self.moving_windows):
            vis.line(
                X=np.asarray(x),
                Y=np.column_stack(y),
                win=m,
                update='replace',
                opts=dict(showlegend=True)
            )


class StaticPlot:
    def __init__(self,
                 parser: Callable[[Sequence[Sequence[float]]], Iterable[float]],
                 title: str,
                 legend: List[str],
                 x_pts: Sequence[float],
                 ylab: str,
                 xlab: str='Time-step'):
        self.parser = parser
        self.title = title
        size = len(x_pts)

        self.y_window = [[0] * size for _ in range(len(legend))]
        self.x_window = list(x_pts)

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

        self.y = [[] for _ in range(len(legend))]

    def update(self, data: dat):
        try:
            self.update_plot(*self.parser(data))
        except:
            print(data)
            print(self.parser(data))
            raise

    def update_plot(self, *args: Sequence):
        for y_list in self.y_window, self.y:
            for y, a in zip(y_list, args):
                y[:] = a

        # Update windowed graph
        vis.line(
            X=np.asarray(self.x_window),
            Y=np.column_stack(self.y_window),
            win=self.moving_window,
            update='replace',
            opts=dict(showlegend=True)
        )


def kanerva_factory(sids: list,
                    actions: List[actn],
                    action_in_state: bool,
                    *args, **kwargs) -> KanervaCoder:
    del args, kwargs

    pos = np.asarray([i / 10 for i in range(-20, 20)])
    speed = np.asarray([i / 10 for i in range(0, 120)])
    load = np.array([-1, 1])

    chunk_size = (len(pos) + len(speed) +
                  (len(actions) if action_in_state else 0))
    counts = np.zeros(load.size * chunk_size)

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
            p, s, l_, *a = xi
            p = np.argmin(np.abs(pos - p))
            s = np.argmin(np.abs(speed - s))
            # p = pos.index(round(p * 10) / 10)
            # s = speed.index(round(s * 10) / 10)
            l_ = int(l_ == 1)

            indices = [l_ * chunk_size + p,
                       l_ * chunk_size + pos.size + s]

            if action_in_state:
                try:
                    a = tuple(a)
                except TypeError:
                    a = (a,)
                indices.append(l_ * chunk_size + pos.size + speed.size +
                               actions.index(a))

            counts[indices] += 1
            indices.append(-1)

            chunks.append(np.asarray(indices))

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
                     gvf2main: mp.SimpleQueue,
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

            # send action and features to GVFs
            gvf_data = (obs, active_pts)
            main2gvf.put(gvf_data)

            # get action control GVFs
            action = gvf2main.get()

            # send action to servos
            instructions = [goal_instruction(a)
                            for a in action
                            if a is not None]
            for sid, instr in zip(sids, instructions):
                await send_msg(sr, sw, sid, instr)

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
                  gvfs: Sequence[Sequence[Learner]],
                  behaviour_gvf: SARSA,
                  main2gvf: mp.SimpleQueue,
                  gvf2main: mp.SimpleQueue,
                  gvf2plot: mp.SimpleQueue):
    action, action_prob, obs, x = None, None, None, None

    # get first state
    while exit_flag.value == 0 and obs is None:
        while exit_flag.value == 0 and main2gvf.empty():
            time.sleep(0.001)
        if exit_flag.value == 0:
            obs, x = main2gvf.get()
            action, action_prob = behaviour_gvf.policy(obs=obs, x=x)
            gvf2main.put(action)

    # main loop
    while exit_flag.value == 0:
        while exit_flag.value == 0 and main2gvf.empty():
            time.sleep(0.001)
        if exit_flag.value:
            break

        # get data from servos
        obsp, xp = main2gvf.get()
        actionp, action_probp = behaviour_gvf.policy(obs=obsp, x=xp)

        # update weights
        for g in chain.from_iterable(gvfs):
            g.update(x, obs,
                     action, action_prob,
                     xp, obsp,
                     actionp, action_probp)

        # send action
        gvf2main.put(actionp)

        # send data to plots
        gdata = [[g.data(x, obs, action, xp, obsp)
                  for g in gs]
                 for gs in gvfs]
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

    try:
        while exit_flag.value == 0:
            while exit_flag.value == 0 and gvf2plot.empty():
                time.sleep(0.001)
            if exit_flag.value:
                break
            data = gvf2plot.get()

            for plot in plots:
                plot.update(data)
    except RuntimeError:
        pass
    finally:
        for plot in plots:
            index = np.arange(len(plot.y[0]))
            np.savetxt(f"{plot.title}.csv",
                       np.column_stack(sum(((np.asarray(y),) for y in plot.y),
                                           (index,))),
                       delimiter=',')


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
    servo = kwargs['servo']
    questn = kwargs['question']
    kwargs['name'] = f'{questn.capitalize()}-{servo}'

    return kwargs


if __name__ == "__main__":
    # servo_loop variables
    servo_ids = [2]
    serial_device = '/dev/tty.usbserial-AI03QEZV'
    m2g = mp.SimpleQueue()
    g2m = mp.SimpleQueue()
    g2p = mp.SimpleQueue()
    eflag = mp.Value('b', False)

    # behaviour policy
    actns = list(product([-1.5, None, 1.5], repeat=1))
    initial_action = (1.5,)

    # controllers
    a_in_s = False
    AC = DiscreteAC if a_in_s else ContinuousAC
    chunksize = (len(list(range(-20, 20)) + list(range(0, 120))) +
                 (len(actns) if a_in_s else 0))
    num_features = chunksize * 2 + 1
    num_active_features = 4 if a_in_s else 3
    kanerva_coder, ks = kanerva_factory(servo_ids,
                                        ptypes=num_features,
                                        num_active=num_active_features,
                                        action_in_state=a_in_s,
                                        actions=actns)
    alpa = 1 / num_active_features
    gamm = 0.8

    # sum of near-in-time positions
    xaxis = np.arange(200) * (3 - -3)/200 + -3

    if a_in_s:
        evals = {'action_probs': Variable('action_probs')}
    else:
        evals = {'pnorm': Normalpdf(xaxis),
                 'μ': Variable('μ'),
                 'σ': Variable('σ')}
    posit = [
             {
              'alpha': alpa,
              'gamma': lambda o: gamm,
              'lamda': 0.9,
              'n_features': num_features,
              'question': 'posit',
              'actions': actns,
              'evaluators': evals
             },
            ]
    cumuls_extreme = [
                   # {'cumulant': lambda x, o, a, xp, op: o[5],
                   #  'servo': 2,
                   #  'sid': 1},
                   {'cumulant': lambda x, o, a, xp, op: o[0],
                    'servo': 1,
                    'sid': 0},
        ]
    cumuls_mod = [{'cumulant': lambda x, o, a, xp, op: -np.abs(o[0] - 0.5),
                   'servo': 1,
                   'sid': 0}
                  ]
    params = product(posit, cumuls_extreme)
    params = product(posit, cumuls_mod)

    # nonsense required to make all the gvfs
    gvf_params = [name_gvf(dict(ChainMap(*p))) for p in params]
    gvf_list = [[AC(**kwargs)
                 for kwargs in gvf_params
                 if i + 1 == kwargs['servo']]
                for i in range(len(servo_ids))]

    # define learning loop
    bg = gvf_list[0][0]
    learning_process = mp.Process(target=learning_loop,
                                  name='learning_loop',
                                  args=(eflag, gvf_list, bg, m2g, g2m, g2p))

    # plotting
    vis = visdom.Visdom()
    vis.close(win=None)

    if a_in_s:
        disc_plots = [Plot(parser=lambda data: data[bg.name]['action_probs'],
                           title='Action probabilities',
                           legend=['-1.5', 'stop', '1.5'],
                           ylab='Probability',
                           sizes=[200])
                      ]
    else:
        cont_plots = [StaticPlot(parser=lambda data: (data[bg.name]['pnorm'],),
                                 title='Distribution',
                                 legend=['Servo 1'],  # , 'Servo 2'],
                                 ylab='Density',
                                 x_pts=xaxis),
                      Plot(parser=lambda data: (data[bg.name]['μ'],
                                                data[bg.name]['σ']),
                           title='Policy parameters',
                           legend=['μ', 'σ'],
                           ylab='Value',
                           sizes=[200])
                      ]

    pos_plot = [
                Plot(parser=lambda data: (data[bg.name]['tderr'],),
                     # , data['obs'][9]),
                     title='TD Error',
                     legend=['Servo 1'],  # , 'Servo 2'],
                     ylab='Error',
                     sizes=[200]),
                Plot(parser=verifier(bg.name,
                                     int(1 / (1 - gamm))*2),
                     # lambda data: (data['obs'][0],),
                     # data['obs'][5]),
                     title='Position',
                     legend=['Predicted', 'Actual'],
                     ylab='Angle',
                     sizes=[100]),
                Plot(parser=lambda data: (data['obs'][4],),
                     # , data['obs'][9]),
                     title='Temp',
                     legend=['Servo 1'],  # , 'Servo 2'],
                     ylab='Temperature',
                     sizes=[200]),
                Plot(parser=lambda data: (data[bg.name]['cumul'],
                                          bg.cumulant([], [0], 0, [], 0)),
                     # , data['obs'][9]),
                     title='Reward',
                     legend=['Servo 1', 'Random'],
                     ylab='Reward',
                     sizes=[200]),
                ]

    plts = pos_plot + (disc_plots if a_in_s else cont_plots)

    plotting_process = mp.Process(target=plotting_loop,
                                  name="plotting",
                                  args=(eflag, g2p, plts))

    # main coroutine setup
    loop = asyncio.get_event_loop()
    coro = servo_loop(device=serial_device,
                      baudrate=1000000,
                      sids=servo_ids,
                      coder=kanerva_coder,
                      main2gvf=m2g,
                      gvf2main=g2m,
                      )
    # coro = data_from_file(main2gvf=m2g, gvf2plot=g2p, coder=kanerva_coder)

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
