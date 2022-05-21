import matplotlib.pyplot as plt
from MathEquations import *
from matplotlib import animation

'''
As the name suggests, this is a class of our model that is using diffusion-reaction equation

Model based on Turing Pattern Equations
∂a(x,t)/∂t= Da Δf(a) + Ra(a(x,t),b(x,t))
∂b(x,t)/∂t= Db Δf(b) + Rb(a(x,t),b(x,t))
t - starting time
dt - increase in time by each iteration
a(x,t) and b(x,t) describe the concentration of chemicals a and b at position x and time t
Ra, Rb - determine how the concentrations change due to interspecies reactions
Da, Db - diffusion coefficients
'''

class RDModel():
    def __init__(self, Da, Db,width, height,dx, dt, steps, alpha, beta):
        self.Da = Da
        self.Db = Db

        self.width = width
        self.height = height
        self.shape = (width, height)
        self.dx = dx
        self.dt = dt
        self.steps = steps
        self.alpha = alpha
        self.beta = beta

    def get_data(self):
        self.t = 1
        self.a, self.b = random_data(self.width, self.height)

    def update(self):
        for i in range(self.steps):
            print(i)
            self.t += self.dt
            La = laplacian2D(self.a, 1)
            Lb = laplacian2D(self.b, 1)

            self.a = self.a + self.dt * (self.Da * La + reactionRA(self.a, self.b, self.alpha))
            self.b = self.b + self.dt * (self.Db * Lb + reactionRB(self.a, self.b, self.beta))

    def draw(self, ax):
        '''plt.cla()
        plt.subplot(1, 2, 1)
        plt.title("A, t = {:.2f}".format(self.t))
        plt.imshow(self.a)

        plt.subplot(1, 2, 2)
        plt.title("A, t = {:.2f}".format(self.t))
        plt.imshow(self.b)'''

        ax[0].clear()
        ax[1].clear()

        ax[0].imshow(self.a, cmap='jet')
        ax[1].imshow(self.b, cmap='copper')

        ax[0].grid()
        ax[1].grid()

        ax[0].set_title("A, t = {:.2f}".format(self.t))
        ax[1].set_title("B, t = {:.2f}".format(self.t))

    def plot_time_evolution(self, filename, n_steps=30):
        self.get_data()
        fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(12, 6))

        def step(frame):
            self.update()
            self.draw(ax)

        anim = animation.FuncAnimation(fig, step, frames=np.arange(n_steps), interval=30)
        anim.save(filename=filename, dpi=60, fps=20, writer='imagemagick')