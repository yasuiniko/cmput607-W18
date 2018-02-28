# coding=utf-8
import Queue
import collections
import multiprocessing as mp

import numpy as np
import visdom

from lib_robotis_hack import *
from rl_glue import BaseAgent, BaseEnvironment, RLGlue


class DanceBot(BaseAgent):
    """Ignores the observations and randomly chooses the 0 or 1 action."""

    def __init__(self, plotting_data):
        self.last_obs = None
        self.last_action = None
        # self.gvfs = gvfs
        self.plotting_data = plotting_data

        super(DanceBot, self).__init__()

    def policy(self, observation):

        action = []
        for i in range(len(observation) / 5):
            if observation[i*5] > 1:
                action.append(-1)
            elif not observation[i * 5] >= -1:
                action.append(1)
            else:
                action.append(self.last_action[i])

        return action

    def agent_start(self, observation):
        predicted = [np.random.uniform() for _ in range(10)]
        self.plotting_data.put((predicted, observation))

        self.last_action = -1, 1
        self.last_obs = observation

        return self.last_action

    def agent_step(self, reward, observation):
        action = self.policy(observation)

        predicted = [np.random.uniform() for _ in range(10)]
        self.plotting_data.put((predicted, observation))

        self.last_obs = observation
        self.last_action = action

        return self.last_action

    def agent_end(self, reward):
        pass


class ServoEnvironment(BaseEnvironment):
    """Single state environment with reward equal to the previous action.
    Terminates after n actions.

    Note: If time_steps is 0 then the termination flag will always be false.
    """

    def __init__(self, servos):
        super(ServoEnvironment, self).__init__()
        self.servos = servos
        self.last_obs = [0]*10

    def observe(self):
        obs = ()
        for i, servo in enumerate(self.servos):
            # obs += (servo.read_angle(),
            #         servo.read_encoder(),
            #         servo.read_voltage(),
            #         servo.read_load(),
            #         servo.read_temperature())

            read_all = [0x02, 0x24, 0x08]
            try:
                data = servo.send_instruction(read_all, servo.servo_id)
                position = (data[0] + data[1] * 256 - 512) * 5 * math.pi / 3069
                speed = (data[2] + data[3] * 256) * 5 * math.pi / 3069
                load = (data[5] & 3) * 128 + data[4]
                load *= math.pow(-1, bool(data[5] & 4))
                voltage = data[6] / 10
                temperature = data[7]
                obs += (position,
                        speed,
                        voltage,
                        load,
                        temperature)
            except RuntimeError:
                obs += tuple(self.last_obs[i*5:i*5+5])

        return obs

    def env_start(self):

        return {'reward': 0,
                'obs': self.observe(),
                'terminal': False}

    def env_step(self, action):
        for i, servo in enumerate(self.servos):
            try:
                servo.move_angle(action[i], blocking=False)
            except RuntimeError:
                raise

        return {'reward': 0,
                'obs': self.observe(),
                'terminal': False}


class Plot:
    def __init__(self, parser, title, ylab, xlab='time-step', size=50):
        self.parser = parser
        self.title = title

        self.actual_results_window = collections.deque(size * [0], size)
        self.predicted_window = collections.deque(size * [0], size)
        self.x_window = collections.deque([0 for _ in range(size)], size)

        self.moving_window = vis.line(
            X=np.column_stack(([0], [0])),
            Y=np.column_stack(([0], [0])),
            opts=dict(
                showlegend=False,
                width=250,
                height=250,
                xlabel=xlab,
                ylabel=ylab,
                title=title,
                marginleft=30,
                marginright=30,
                marginbottom=80,
                margintop=30,
                legend=['Predicted', 'Actual'],
            ),
        )

        self.i = 0
        self.size = size
        self.actual = []
        self.predicted = []

    def update(self, data):
        self.update_plot(*self.parser(data))

    def update_plot(self, predicted, actual):
        self.i += 1

        self.actual_results_window.append(actual)
        self.predicted_window.append(predicted)
        self.x_window.append(self.i)

        # Update windowed graph
        vis.line(
            X=np.column_stack((self.x_window, self.x_window)),
            Y=np.column_stack(
                (self.predicted_window,
                 self.actual_results_window)),
            win=self.moving_window,
            update='replace'
        )

        self.actual.append(actual)
        self.predicted.append(predicted)


def plotting(exit_flag, plotting_data, plots):
    while exit_flag.value == 0:
        data = None
        while exit_flag.value == 0 and data is None:
            try:
                data = plotting_data.get(block=True, timeout=0.01)
            except Queue.Empty:
                pass

        for plot in plots:
            plot.update(data)

    for plot in plots:
        actual = np.asarray(plot.actual)
        predicted = np.asarray(plot.predicted)
        index = np.arange(actual.size)
        np.savetxt("{}.csv".format(plot.title),
                   (index, predicted, actual),
                   delimiter=',')


def find_servos(dyn):
    """Finds all servo IDs from 0 to 5 on the USB2Dynamixel"""
    print 'Scanning for Servos.'
    servos = []
    dyn.servo_dev.setTimeout(0.03)  # To make the scan faster
    for j in range(5):
        try:
            Robotis_Servo(dyn, j)
            print("\n FOUND A SERVO @ ID {}\n".format(j))
            servos.append(j)
        except RuntimeError:
            pass
    dyn.servo_dev.setTimeout(1.0)  # Restore to original
    return servos


def foreground(servos, time_steps, episodes, plotting_data):
    my_agent = DanceBot(plotting_data)
    my_env = ServoEnvironment(servos)
    my_rl_glue = RLGlue(my_env, my_agent)

    print("\nRunning {} episodes with {} time-steps each.".format(episodes,
                                                                  time_steps))
    for _ in range(episodes):
        my_rl_glue.rl_episode(time_steps)


if __name__ == '__main__':
    vis = visdom.Visdom()
    vis.close(win=None)

    d = USB2Dynamixel_Device(dev_name="/dev/tty.usbserial-AI03QEZV",
                             baudrate=1000000)
    # make sure the servo IDs are unique before running this, using
    # twitch_init.ipynb
    s_list = find_servos(d)
    servos = [Robotis_Servo(d, s) for s in s_list]
    episodes = int(sys.argv[1])
    time_steps = int(sys.argv[2])
    q = mp.Queue()
    exit_flag = mp.Value('b', False)

    plots = []
    topics = ["position (rad)",
              "speed (rad/s)",
              "voltage (V)",
              "load (raw)",
              "temperature (°C)"]

    # # for plotting a predicted and actual value for each servo*topic
    # for servo_ind in range(len(servos)):
    #     for topic in topics:
    #         title = "Servo {} {}".format(servo_ind + 1, topic.split()[0])
    #         ylab = topic
    #         c_ind = int(servo_ind * len(topics) + topics.index(topic))
    #
    #         # for loop iterations share scope, so we have to use another
    #         # closure to pass c_ind properly into our parser
    #         plots.append(Plot(title=title,
    #                           ylab=topic,
    #                           parser=(lambda c_ind:
    #                                   lambda x: [x[0][c_ind],
    #                                              x[1][c_ind]])(c_ind)))

    # for plotting both servos in one plot
    for topic in topics:
        title = topic.split()[0].title()
        ylab = topic

        # for loop iterations share scope, so we have to use another
        # closure to pass c_ind properly into our parser
        plots.append(Plot(title=title,
                          ylab=topic,
                          parser=(lambda topic:
                                  lambda x: [x[1][int(topics.index(topic))],
                                             x[1][int(len(topics) +
                                                      topics.index(
                                                          topic))]])(topic)))

    plotting_process = mp.Process(target=plotting,
                                  name="plotting",
                                  args=(exit_flag, q, plots))

    foreground_process = mp.Process(target=foreground,
                                    name="foreground",
                                    args=(servos,
                                          time_steps,
                                          episodes,
                                          q))

    try:
        # plotting_process.start()
        # foreground(servos, time_steps, episodes, q)
        foreground_process.start()
        foreground_process.join()
    except KeyboardInterrupt:
        pass
    finally:
        exit_flag.value = 1
        plotting_process.join()

        # "Success" Dance
        for i in reversed(range(5)):
            for j in [-1, 1]:
                for s in servos:
                    s.move_to_encoder(512 + 30 * i * j)
                while any(s.is_moving() for s in servos):
                    time.sleep(0.01)

        # Disable torque
        for s in servos:
            s.disable_torque()
