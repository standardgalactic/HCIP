import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


class TwoArms:
    def __init__(self, params):
        self.width = params['width']
        self.height = params['height']
        self.n_channels = params['n_channels']
        self.len_pillar = params['len_pillar']
        self.c_pillar = params['c_pillar']
        self.len_arm1 = params['len_arm1']
        self.len_arm2 = params['len_arm2']
        self.c_arms = params['c_arms']
        self.c_ball = params['c_ball']
        self.D1 = params['D1']
        self.velocity = params['velocity']

        self.center_x = int(np.floor(self.height / 2))
        self.center_y = int(np.floor(self.width / 2))

    def drawTrial(self):
        time_travel1 = int(self.len_pillar/self.velocity)
        if self.D1 == 0:
            time_travel2 = int(self.len_arm1/self.velocity)
        else:
            time_travel2 = int(self.len_arm2/self.velocity)
        time_travel_total = time_travel1 + time_travel2
        images = np.zeros((time_travel_total, self.width, self.height, self.n_channels))

        pos_ball = np.zeros((time_travel_total, 2))
        for i in np.arange(time_travel_total):
            images[i, :, :, :] = self.drawMaze()
            if 0 <= i < time_travel1:
                images[i, self.center_x-self.len_pillar+i+1, self.center_y, :] = self.c_ball
            elif time_travel1 <= i < time_travel_total:
                images[i, self.center_x, self.center_y+(self.D1*2-1)*(i-time_travel1+1), :] = self.c_ball

        im = plt.imshow(images[5, :, :, :], vmin=0, vmax=1)
        ax = im.axes
        ax.set_xticks(np.arange(0, self.width, 1))
        ax.set_yticks(np.arange(0, self.height, 1))
        ax.grid()
        plt.show()
        a = 0

    def drawMaze(self):
        center_x = self.center_x
        center_y = self.center_y

        image = np.zeros((self.width, self.height, self.n_channels))

        # draw pillar
        image[center_x - self.len_pillar + 1:center_x + 1, center_y, :] = self.c_pillar

        # draw arms
        image[center_x, center_y + 1:center_y + self.len_arm2 + 1, :] = self.c_arms
        image[center_x, center_y - self.len_arm1:center_y, :] = self.c_arms

        return image
        # # draw ball
        # image[center_x, center_y, :] = self.c_ball
        #
        # im = plt.imshow(image, vmin=0, vmax=1)
        # ax = im.axes
        # ax.set_xticks(np.arange(0, self.width, 1))
        # ax.set_yticks(np.arange(0, self.height, 1))
        # ax.grid()
        # plt.show()


params = {
    'width': 23,
    'height': 23,
    'n_channels': 3,  # RGB
    'len_pillar': 5,
    'c_pillar': (.7, .7, .7),
    'len_arm1': 5,
    'len_arm2': 10,
    'c_arms': (1, 1, 1),
    'c_ball': [.8, .3, .3],
    'D1': 1,
    'velocity': 1
}

drawer = TwoArms(params)
conv2d = nn.Conv2d(3, 5, 3)
a = 0
# drawer.drawTrial()
