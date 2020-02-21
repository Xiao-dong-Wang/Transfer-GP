import autograd.numpy as np
import sys
import toml
from util import *
from TGP import TGP
from get_dataset import *
import multiprocessing
import pickle
import matplotlib.pyplot as plt

np.random.seed(9)

argv = sys.argv[1:]
conf = toml.load(argv[0])

name = conf['funct']
funct = get_funct(name)
num = conf['num']
bounds = np.array(conf['bounds'])
bfgs_iter = conf['bfgs_iter']

#### TGP
dataset = init_dataset(funct, num, bounds)
src_x = dataset['src_x']
src_y = dataset['src_y']
tag_x = dataset['tag_x']
tag_y = dataset['tag_y']
model = TGP(dataset, bfgs_iter[0], debug=True)
model.train()



# Test data
nn = 200
X_star = np.linspace(-0.5, 0.5, nn)[None,:]
y_star_tag = funct[1](X_star,bounds)
y_star_src = funct[0](X_star,bounds)
X_star_real = X_star * (bounds[0,1]-bounds[0,0]) + (bounds[0,1]+bounds[0,0])/2
y_pred, y_var = model.predict(X_star)

src_x_real = src_x * (bounds[0,1]-bounds[0,0]) + (bounds[0,1]+bounds[0,0])/2
tag_x_real = tag_x * (bounds[0,1]-bounds[0,0]) + (bounds[0,1]+bounds[0,0])/2


plt.figure()
plt.cla()
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)
plt.plot(X_star_real.flatten(), y_star_tag.flatten(), 'b-', label = "tag function", linewidth=2)
plt.plot(X_star_real.flatten(), y_star_src.flatten(), 'g-', label = "src function", linewidth=2)
plt.plot(X_star_real.flatten(), y_pred.flatten(), 'r--', label = "Prediction", linewidth=2)
lower = y_pred - 2.0*np.sqrt(y_var)
upper = y_pred + 2.0*np.sqrt(y_var)
plt.fill_between(X_star_real.flatten(), lower.flatten(), upper.flatten(), 
                 facecolor='pink', alpha=0.5, label="Two std band")
plt.plot(src_x_real, src_y, 'go')
plt.plot(tag_x_real, tag_y, 'ko')
plt.legend()
ax = plt.gca()
ax.set_xlim([bounds[0,0], bounds[0,1]])
plt.xlabel('x')
plt.ylabel('f(x)')

plt.show()






