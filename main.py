from ReactionDiffusionModel import *

#rybka
width = 200
dx = 1
dt = 0.001

x,y = np.mgrid[0:width,0:width]
Da, Db, alpha, beta = 1, 100, 0.01, 10

beta = np.random.rand(width,width)


RDModel = RDModel(Da, Db,width=width, height=width,dx=dx, dt=dt, steps=100, alpha=alpha, beta=beta)
RDModel.plot_time_evolution("test.gif", n_steps=150)

#panterka
#cm=copper
'''width = 200
dx = 1
dt = 0.001

Da, Db, alpha, beta = 1, 100, 0.01, 0.25

RDModel = RDModel(Da, Db,width=width, height=width,dx=dx, dt=dt, steps=100, alpha=alpha, beta=beta)
RDModel.plot_time_evolution("test.gif", n_steps=150)'''

'''width = 200
dx = 1
dt = 0.001

Da, Db, alpha, beta = 1, 100, 0.01, 0.25

RDModel = RDModel(Da, Db,width=width, height=width,dx=dx, dt=dt, steps=100, alpha=alpha, beta=beta)
RDModel.plot_time_evolution("2dRD.gif", n_steps=150)'''