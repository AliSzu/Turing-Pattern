import diffusion_reaction
from diffusion_reaction import *

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
class model():
    def __init__(self, Da, Db, Ra, Rb, t, dt, shape, alpha, beta):
        self.Db = Db
        self.Da = Da
        self.Ra = Ra
        self.Rb = Rb
        self.t = t
        self.dt = dt
        self.height = shape
        self.width = shape
        self.alpha = alpha
        self.beta = beta

    def get_data(self):
        self.a, self.b = diffusion_reaction.random_data(self.height, self.width)

    def update(self):
        La = diffusion_reaction.laplacian2D(self.a)
        Lb = diffusion_reaction.laplacian2D(self.b)

        #To calculate the next argument we are going to use Forawrd Euler method
        #a(x, t+1) = a(x,t) + ∂a(x,t)/∂t * delta_t

        self.a = self.a + (self.Da * La + diffusion_reaction.reactionRA(self.a, self.b, self.alpha))
        self.b = self.b + (self.Db * Lb + diffusion_reaction.reactionRB(self.a, self.b, self.beta))



