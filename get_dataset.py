import autograd.numpy as np
import os
import string

def init_dataset(funct, num, bounds):
    dim = bounds.shape[0]
    num_src = num[0]
    num_tag = num[1]
    src_x = np.random.uniform(-0.5, 0.5, (dim, num_src))
    tag_x = np.random.uniform(-0.5, 0.5, (dim, num_tag))

    dataset = {}
    dataset['src_x']    = src_x
    dataset['tag_x']    = tag_x
    dataset['src_y']    = funct[0](src_x, bounds)
    dataset['tag_y']    = funct[1](tag_x, bounds)
    return dataset

def get_test(funct, num, bounds):
    dim = bounds.shape[1]
    dataset = {}
    dataset['test_x'] = np.random.uniform(-0.5, 0.5, (dim, num))
    dataset['test_y'] = funct[1](dataset['test_x'], bounds)
    return dataset


# bounds:  -0.5 : 1
def test1_tag(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    ret = (x+0.03)**2 * np.sin(5.0*np.pi*(x+0.03))+0.1
    return ret.reshape(1, -1)

def test1_src(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    ret = x**2 * np.sin(5.0*np.pi*x)
    return ret.reshape(1, -1)


# bounds:   0 : 1
def test2_tag(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    ret = (6.0*x - 2.0)**2 * np.sin(12.*x - 4.0)
    return ret.reshape(1, -1)

def test2_src(x, bounds):
    tmp = test2_tag(x, bounds)
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    ret = 0.5*tmp + 10.0*(x-0.5) - 5.0
    return ret.reshape(1, -1)

def get_funct(funct):
    if funct == 'test1':
        return [test1_src, test1_tag]
    elif funct == 'test2':
        return [test2_src, test2_tag]
    else:
        return [test1_src, test1_tag]

