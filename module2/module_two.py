"""Kanerva coding code is modified from code by Kris de Asis.
Plotting code is modified from code by Cam Linke.
Everything else by Niko Yasui.

Use Python 3.5 or higher.
"""
import asyncio
import collections
import math
import multiprocessing as mp
import time
from functools import partial
from typing import Callable, Iterable, List, NewType, Sequence, Tuple, TypeVar

import numpy as np
import serial_asyncio
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
    position = (data[0] + data[1] * 256 - 512) * 5 * math.pi / 3069
    speed = (data[2] + data[3] * 256) * 5 * math.pi / 3069
    load = (data[5] & 3) * 256 + data[4] * (1 - 2 * bool(data[5] & 4))
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
actn = TypeVar('action')
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


class Learner:
    def __init__(self,
                 alpha: float,
                 gamma: Callable[[obsv], float],
                 cumulant: Callable[[np.ndarray, obsv, actn, np.ndarray, obsv],
                                    float],
                 lambda_: float):
        self.alpha = alpha * (1 - lambda_)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.cumulant = cumulant


class TDLearner(Learner):
    def __init__(self,
                 alpha: float,
                 gamma: Callable[[obsv], float],
                 cumulant: Callable[[np.ndarray, obsv, actn, np.ndarray, obsv],
                                    float],
                 lambda_: float):
        super().__init__(alpha, gamma, cumulant, lambda_)


class GTDLearner(Learner):
    def __init__(self,
                 alpha: float,
                 gamma: Callable[[obsv], float],
                 cumulant: Callable[[np.ndarray, obsv, actn, np.ndarray, obsv],
                                    float],
                 lambda_: float,
                 target_policy: DiscretePolicy,
                 n_features: int,
                 beta_0: float=None,
                 alpha_h: float=None,
                 alpha_h_hat: float=None):
        super().__init__(alpha, gamma, cumulant, lambda_)

        # GTD
        self.alpha_h = self.alpha / 100 if alpha_h is None else alpha_h
        self.w = np.zeros(n_features)
        self.e = np.zeros(n_features)
        self.h = np.zeros(n_features)
        self.la = initial_action
        self.target_policy = target_policy

        # RUPEE, see Adam White's 2015 thesis p.132, 136, 138, 142
        self.alpha_h_hat = (10 * self.alpha
                            if alpha_h_hat is None
                            else alpha_h_hat)
        self.beta0_r = self.alpha / 50 if beta_0 is None else beta_0
        self.h_hat = np.zeros(n_features)
        self.d_e = np.zeros(n_features)
        self.t = 0

        # UDE
        self.sample_mean = 0
        self.ewma = 0
        self.n = 0
        self.M2 = 0
        self.beta0_u = self.alpha * 10 if beta_0 is None else beta_0

    def update(self,
               action: actn,
               action_prob: float,
               obs: obsv,
               obsp: obsv,
               x: np.ndarray,
               xp: np.ndarray):

        # for readability
        alpha, lambda_, beta0_u = self.alpha, self.lambda_, self.beta0_u
        w, e, h = self.w, self.e, self.h
        t, h_hat, d_e, beta0_r = self.t, self.h_hat, self.d_e, self.beta0_r

        # precalculations
        gammap = self.gamma(obsp)
        rho = self.target_policy.prob(w=w, x=x, obs=obs, a=action) / action_prob
        reward = self.cumulant(x, obs, action, xp, obsp)
        delta = reward + gammap * w[xp].sum() - w[x].sum()
        assert rho == 1 or self.alpha_h > 0

        # gtd
        e *= rho * lambda_ * self.gamma(obs)
        e[x] += rho
        delta_e = delta * e
        w[xp] += alpha * (delta_e[xp] - gammap * (1 - lambda_) * np.dot(e, h))
        h[x] += self.alpha_h * (delta_e[x] - h[x].sum())

        # rupee
        t = (1 - beta0_r) * t + beta0_r
        h_hat[x] += self.alpha_h_hat * (delta_e[x] - h_hat[x].sum())
        d_e = ((1 - beta0_r / t) * d_e + beta0_r / t * delta_e)

        # ude
        old_mean = self.sample_mean
        self.n += 1
        self.sample_mean += (delta - self.sample_mean) / self.n
        self.M2 += (delta - old_mean) * (delta - self.sample_mean)
        self.ewma = (1 - beta0_u / t) * self.ewma + (beta0_u / t) * delta

        # reassign variables
        self.w, self.e, self.h = w, e, h
        self.t, self.h_hat, self.d_e = t, h_hat, d_e

    @property
    def rupee(self) -> float:
        return np.sqrt(np.abs(np.dot(self.h_hat, self.d_e)))

    @property
    def ude(self) -> float:
        return np.abs(self.ewma / (np.sqrt(self.M2 / self.n) + 0.0000001))

    def predict(self, x: np.ndarray) -> float:
        return self.w[x].sum() + self.w[-1]

    def data(self,
             x: np.ndarray,
             obs: obsv,
             action: actn,
             xp: np.ndarray,
             obsp: obsv) -> List[float]:
        return [self.rupee,
                self.ude,
                self.predict(x),
                self.cumulant(x, obs, action, xp, obsp)]


class Plot:
    def __init__(self,
                 parser: Callable[[Sequence[Sequence[float]]], Iterable[float]],
                 title: str,
                 legend: List[str],
                 ylab: str,
                 xlab: str='time-step',
                 size: int=50):
        self.parser = parser
        self.title = title

        self.y_windows = [collections.deque(size * [0], size)
                          for _ in range(len(legend))]
        self.predicted_window = collections.deque(size * [0], size)
        self.x_window = collections.deque([0 for _ in range(size)], size)

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
        self.update_plot(*self.parser(data))

    def update_plot(self, *args: float):
        self.i += 1

        for y_list in self.y_windows, self.y:
            for y, a in zip(y_list, args):
                y.append(a)
        self.x_window.append(self.i)

        # Update windowed graph
        vis.line(
            X=np.asarray(self.x_window),
            Y=np.column_stack(self.y_windows),
            win=self.moving_window,
            update='replace',
            opts=dict(showlegend=True)
        )


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

    def __call__(self, x):
        xs = (x - self._lims[:, 0]) / self._ranges
        near = np.argpartition(np.max(np.abs(self._pts - xs), axis=1), self._k)
        return near[:self._k]


def kanerva_factory(sids: list, ptypes, num_active, action_in_state):
    dims = (5 + action_in_state) * len(sids)

    # value limits of each dimension (min, max)
    obs_lims = [(-1.7, 1.7),  # angular position (rads)
                (0, 12),  # speed (rads/s)
                (-300, 300),  # load (unknown)
                (0, 12),  # voltage (V)
                (-10, 90)]  # temperature (C)
    action_lims = [(-1, 1)]

    # one for each servo
    lims = sum([obs_lims for _ in sids], [])
    lims += sum([action_lims for _ in sids if action_in_state], [])

    # create kanervacoder
    kan = KanervaCoder(dims, ptypes - 1, (num_active - 1) / (ptypes - 1), lims)

    # add bias unit
    def coder(x):
        return np.append(kan(x), [-1])

    return coder


async def data_from_file(main2gvf: mp.SimpleQueue,
                         gvf2plot: mp.SimpleQueue,
                         coder: KanervaCoder):
    data = np.load('offline_data.npy')

    for item in data:
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
    for sid in sids:
        await send_msg(sr, sw, sid, [0x03, 0x20, 0x00, 0x01])

    # set initial action
    action = initial_action

    # some constants
    read_data = [0x02,  # read
                 0x24,  # starting from 0x24
                 0x08]  # a string of 8 bytes

    # hi = [1.7, 12, 300, 12, 90]
    # lo = [-1.7, 0, -300, 0, -10]
    # observation = np.random.random(5)
    # for i in range(observation.size):
    #     observation[i] = observation[i] * (hi[i] - lo[i]) + lo[i]
    # observation = list(observation)

    store_data = []

    try:
        for _ in range(5000):
            # read data from servos
            byte_data = [await send_msg(sr, sw, sid, read_data) for sid in sids]

            # convert to human-readable data
            obs = sum([parse_data(bd) for bd in byte_data], []) + list(action)

            # get active tiles in kanerva coding
            active_pts = coder(obs)

            # get most recent weights from control GVFs
            pass

            # decide on an action
            action, action_prob = behaviour_policy(obs=obs)

            # send action and features to GVFs
            gvf_data = (action, action_prob, obs, active_pts)
            main2gvf.put(gvf_data)

            # record data for later
            store_data.append(gvf_data)

            # send action to servos
            instructions = [goal_instruction(a) for a in action]
            for sid, instr in zip(sids, instructions):
                await send_msg(sr, sw, sid, instr)

        np.save('offline_data.npy', store_data)

    except KeyboardInterrupt:
        pass
    finally:
        sr.read()
        await sw.drain()

        for sid in sids:
            write(sw, sid, [0x03, 0x18, 0x00])  # disable torque


def learning_loop(exit_flag: mp.Value,
                  gvfs: Sequence[GTDLearner],
                  main2gvf: mp.SimpleQueue,
                  gvf2plot: mp.SimpleQueue):
    action, action_prob, obs, x = None, None, None, None

    # get first state
    while exit_flag.value == 0 and obs is None:
        while exit_flag.value == 0 and main2gvf.empty():
            time.sleep(0.001)
        if exit_flag.value == 0:
            action, action_prob, obs, x = main2gvf.get()

    # main loop
    while exit_flag.value == 0:
        while exit_flag.value == 0 and main2gvf.empty():
            time.sleep(0.001)
        if exit_flag.value:
            break

        # get data from servos
        actionp, action_probp, obsp, xp = main2gvf.get()

        # update weights
        for g in gvfs:
            g.update(action, action_prob, obs, obsp, x, xp)

        # send data to plots
        data = [[obs]] + [g.data(x, obs, action, xp, obsp) for g in gvfs]
        gvf2plot.put(data)

        # go to next state
        obs = obsp
        x = xp
        action = actionp
        action_prob = action_probp


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
                   sum(((np.asarray(y),) for y in plot.y), (index,)),
                   delimiter=',')


def waver(actions: Sequence[actn],
          obs: obsv,
          *args, **kwargs) -> Sequence[float]:
    action = []
    n = len(obs) // 5
    last_action = obs[-n:]

    # loop through each servo's 5-tuple
    for i in range(n):
        pos = obs[i * 5]
        if pos > 1:
            action.append(-1)
        elif pos < -1:
            action.append(1)
        else:
            action.append(last_action[i])

    index = actions.index(action)
    probs = [0] * len(actions)
    probs[index] = 1

    return probs


def turner(actions: Sequence[actn],
           obs: obsv,
           *args, **kwargs) -> Sequence[float]:
    a0 = 1 if obs[5] >= 0 else -1

    return [1 if action[0] == a0 else 0 for action in actions]


def discounted_sum(gammas: Sequence[float],
                   x: Sequence[float]):
    sm = x[0]
    gam = 1

    for i in range(1, min(len(gammas), len(x)) - 1):
        gam *= gammas[i]
        sm += x[i] * gam

    return sm


def verifier(gvf_gamma: Callable[[obsv], float],
             gvf_index: int,
             wait_time: int) -> Callable[[dat], Tuple[float, float]]:
    predictions = collections.deque([], wait_time)
    cumulants = collections.deque([], wait_time)
    gammas = collections.deque([], wait_time)

    def f(data: dat) -> Tuple[float, float]:
        predictions.append(data[gvf_index + 1][2])
        cumulants.append(data[gvf_index + 1][3])
        gammas.append(gvf_gamma(data[0][0]))

        return predictions[0], discounted_sum(gammas, cumulants)

    return f


if __name__ == "__main__":
    # servo_loop variables
    servo_ids = [2, 3]
    serial_device = '/dev/tty.usbserial-AI03QEZV'
    m2g = mp.SimpleQueue()
    m2p = mp.SimpleQueue()
    g2p = mp.SimpleQueue()
    eflag = mp.Value('b', False)

    # behaviour policy
    wave_actions = [[1, 1], [-1, 1], [1, -1], [-1, -1]]
    initial_action = [1, -1]
    wave = DiscretePolicy(wave_actions, partial(waver, wave_actions))

    # gvfs
    num_features = 1025
    num_active_features = 33
    wt_policy = DiscretePolicy(wave_actions, partial(turner, wave_actions))
    kanerva_coder = kanerva_factory(servo_ids,
                                    ptypes=num_features,
                                    num_active=num_active_features,
                                    action_in_state=True)

    # time to turn if always going towards the closer of -1 or 1
    off_time_turn = GTDLearner(alpha=0.01 / num_active_features,
                               gamma=lambda o: 1 if -1 < o[5] < 1 else 0,
                               lambda_=0.9,
                               cumulant=lambda x, o, a, xp, op: 1,
                               target_policy=wt_policy,
                               n_features=num_features)

    # on-policy time until hitting -1 or 1
    on_time_turn = GTDLearner(alpha=0.1 / num_active_features,
                              gamma=lambda o: 1 if -1 < o[5] < 1 else 0,
                              lambda_=0.9,
                              cumulant=lambda x, o, a, xp, op: 1,
                              target_policy=wave,
                              n_features=num_features,
                              alpha_h=0)

    # next position on-policy
    on_next_pos = GTDLearner(alpha=0.1 / num_active_features,
                             gamma=lambda o: 0,
                             lambda_=0.9,
                             cumulant=lambda x, o, a, xp, op: op[0],
                             target_policy=wave,
                             n_features=num_features,
                             alpha_h=0)
    gvf_list = off_time_turn, on_next_pos, on_time_turn
    learning_process = mp.Process(target=learning_loop,
                                  name='learning_loop',
                                  args=(eflag, gvf_list, m2g, g2p))

    # plotting
    vis = visdom.Visdom()
    vis.close(win=None)
    ntt_parser = verifier(on_time_turn.gamma, gvf_list.index(on_time_turn), 50)
    nnp_parser = verifier(on_next_pos.gamma, gvf_list.index(on_next_pos), 1)
    plts = [Plot(parser=lambda data: data[1][2:3],
                 title='Off-policy time to extreme position',
                 legend=['Predicted'],
                 ylab='Value'),
            Plot(parser=lambda data: data[1][:2],
                 title='Error for off-policy time to extreme position',
                 legend=['RUPEE', 'UDE'],
                 ylab='Error'),
            Plot(parser=ntt_parser,
                 title='On-policy time to extreme position',
                 legend=['Predicted', 'Actual'],
                 ylab='Value'),
            Plot(parser=nnp_parser,
                 title='On-policy next position',
                 legend=['Predicted', 'Actual'],
                 ylab='Value',
                 size=100),
            ]

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
                      behaviour_policy=wave)
    # coro = data_from_file(main2gvf=m2g, gvf2plot=g2p, coder=kanerva_coder)

    # start processes
    plotting_process.start()
    learning_process.start()

    # main loop
    try:
        loop.run_until_complete(coro)
    except KeyboardInterrupt:
        pass
    finally:
        eflag.value = 1
        plotting_process.join()
        learning_process.join()
        loop.close()
