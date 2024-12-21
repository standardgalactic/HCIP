import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class Arms:
    def __init__(self, params):
        self.len_arm1 = params['len_arm1']
        self.len_arm2 = params['len_arm2']
        self.len_arm3 = params['len_arm3']
        self.len_arm4 = params['len_arm4']
        self.len_arm5 = params['len_arm5']
        self.len_arm6 = params['len_arm6']
        self.width_arm = params['width_arm']

    def plot(self):
        len_arm1 = self.len_arm1
        len_arm2 = self.len_arm2
        len_arm3 = self.len_arm3
        len_arm4 = self.len_arm4
        len_arm5 = self.len_arm5
        len_arm6 = self.len_arm6
        width = self.width_arm
        len_pillar = 10
        fig, ax = plt.subplots(1, figsize=(6, 6))
        print(fig.dpi)
        pillar = Rectangle((-width/2, -width/2), width, len_pillar, color=(.7, .7, .7))
        ax.add_patch(pillar)

        arm1 = Rectangle((-len_arm1-width/2, -width/2), len_arm1, width, color=(.4, .4, .4))
        ax.add_patch(arm1)

        arm2 = Rectangle((width/2, -width/2), len_arm2, width, color=(.4, .4, .4))
        ax.add_patch(arm2)

        arm3 = Rectangle((-len_arm1 - width / 2, width / 2), width, len_arm3, color=(.2, .2, .2))
        ax.add_patch(arm3)
        arm4 = Rectangle((-len_arm1 - width / 2, -len_arm4-width / 2), width, len_arm4, color=(.2, .2, .2))
        ax.add_patch(arm4)

        arm5 = Rectangle((len_arm2 - width / 2, width / 2), width, len_arm5, color=(.2, .2, .2))
        ax.add_patch(arm5)
        arm6 = Rectangle((len_arm2 - width / 2, -len_arm6-width / 2), width, len_arm6, color=(.2, .2, .2))
        ax.add_patch(arm6)

        plt.axis('off')

        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        plt.show()
        fig.savefig('maze.png')



params = {
    'len_arm1': 10,
    'len_arm2': 10,
    'len_arm3': 10,
    'len_arm4': 10,
    'len_arm5': 10,
    'len_arm6': 10,
    'width_arm': 1
}
arms = Arms(params)
arms.plot()




