import numpy as np

'''
X , Y - shape of an array
These are random data that we will work on. Nothing more
'''
def random_data(X,Y):
    array = (np.random.rand(X,Y),np.random.rand(X,Y))
    return array

'''
A - array with our data

Δf (p) - Laplacian symbol
The Diffusion-Reaction model used by Turing looks like this

∂a(u)/∂t=D*Δf(u) + γ*f(u,v)
∂a(v)/∂t=D*Δf(v) + γ*g(u,v)

u=(x,t) where x is a position and t is time

In this part we are interested in the diffusion part that is
D*Δf(u)
The diffusion is obtained by a laplace function. 
It can be computed by summing over neighboring cells and subtract the value of the original cell with the total weight
source: https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Implementation_via_operator_discretization
Because we are working on a 2D array our laplace formula will look like this:

Lets assume that u = (x,y)
ΔfA(x,y) = A(x,y-1) + A(x,y+1) + A(x-1,y) + A(x+1,y) - 4 * A(x,y)
The formula is based on the matrix that was found on the wikipedia page

Because we need to add each neighbour to every cell we have used numpy.roll that shifts entire array
'''

def laplacian2D(A):
    array = np.roll(A,1,axis=1)  #add right
    array += np.roll(A,-1,axis=1)  #add left
    array += np.roll(A,1,axis=0) #add top
    array += np.roll(A,-1,axis=0)  #add bottom
    array +=  -4*A #add middle
    return array

'''
a,b - arrays
This will calculate the reaction part that look like this
Ra(a(x,t),b(x,t))

To calculate the Ra/Rb there are few different equations. This project will use the FitzHugh–Nagumo equation
that looks like this
Ra(a,b) = a - a*a*a - b + alpha
Rb(a,b) = beta(a-b)
Where alpha and beta are constants
'''

def reactionRA(a,b, alpha):
    return a - a*a*a - b + alpha

def reactionRB(a,b, beta):
    return beta*(a-b)