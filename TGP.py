import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import fmin_l_bfgs_b
from util import chol_inv
import traceback
import sys

# Transfer Gaussian Process
class TGP:
    # Initialize TGP class
    # train_x shape: (dim_in, num_train);   train_y shape: (dim_out, num_train) 
    def __init__(self, dataset, bfgs_iter=2000, debug=True):
        self.src_x = dataset['src_x']
        self.src_y = dataset['src_y']
        self.tag_x = dataset['tag_x']
        self.tag_y = dataset['tag_y']
        self.train_x = np.hstack((self.src_x, self.tag_x))
        self.train_y = np.hstack((self.src_y, self.tag_y))
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.dim = self.tag_x.shape[0]
        self.num_src = self.src_x.shape[1]
        self.num_tag = self.tag_x.shape[1]
        self.jitter = 1e-4
        self.normalize()
#        self.lamd = 1

    # Normalize y
    def normalize(self):
        self.train_y = self.train_y.reshape(-1)
        self.mean = self.train_y.mean()
        self.std = self.train_y.std() + 0.000001
        self.train_y = (self.train_y - self.mean)/self.std

        self.src_y = self.train_y[:self.num_src]
        self.tag_y = self.train_y[self.num_src:]

    # Initialize hyper_parameters
    #   theta: output_scale, length_scale, sigma2_src, sigma2_tag, lambda
    def get_default_theta(self):
        theta = np.random.randn(4 + self.dim)
        for i in range(self.dim):
            theta[1+i] = np.maximum(-100, np.log(0.5*(self.train_x[i].max() - self.train_x[i].min()))) #length scale
        theta[self.dim+1] = np.log(np.std(self.src_y)) # sigma2_src
        theta[self.dim+2] = np.log(np.std(self.tag_y)) # sigma2_tag
        theta[self.dim+3] = 2 * np.random.random(1) - 1 # -1< lambda <1
        return theta

    # inner domain kernel
    def kernel1(self, x, xp, theta):
        output_scale = np.exp(theta[0])
        lengthscales = np.exp(theta[1:self.dim+1]) + 0.000001
        diffs = np.expand_dims((x.T/lengthscales).T, 2) - np.expand_dims((xp.T/lengthscales).T, 1)
        return output_scale * np.exp(-0.5*np.sum(diffs**2, axis=0))
    
    # inter domain kernel
    def kernel2(self, x, xp, theta):
        lamd = theta[self.dim+3]
        return lamd * self.kernel1(x, xp, theta)

    def kernel(self, src_x, tag_x, theta):
        # K =
        # K_ss  K_st
        # K_ts  K_tt
        sigma2_src = np.exp(theta[self.dim+1])
        sigma2_tag = np.exp(theta[self.dim+2])
        K_ss = self.kernel1(src_x, src_x, theta) + sigma2_src * np.eye(self.num_src) + self.jitter*np.eye(self.num_src)
        K_st = self.kernel2(src_x, tag_x, theta)
        K_ts = K_st.T
        K_tt = self.kernel1(tag_x, tag_x, theta) + sigma2_tag * np.eye(self.num_tag) + self.jitter*np.eye(self.num_tag)
        tmp1 = np.hstack((K_ss, K_st))
        tmp2 = np.hstack((K_ts, K_tt))
        K = np.concatenate((tmp1, tmp2))
        return K

    def neg_log_likelihood(self, theta):
#        K = self.kernel(self.src_x, self.tag_x, theta)
#        L = np.linalg.cholesky(K)
#        logDetK = np.sum(np.log(np.diag(L)))
#        alpha = chol_inv(L, self.train_y.T)
#        nlz = 0.5*(np.dot(self.train_y, alpha) + (self.num_src+self.num_tag) * np.log(2*np.pi)) + logDetK
        sigma2_src = np.exp(theta[self.dim+1])
        sigma2_tag = np.exp(theta[self.dim+2])
        K_ss = self.kernel1(self.src_x, self.src_x, theta) + sigma2_src * np.eye(self.num_src) + self.jitter*np.eye(self.num_src)
        K_st = self.kernel2(self.src_x, self.tag_x, theta)
        K_ts = K_st.T
        K_tt = self.kernel1(self.tag_x, self.tag_x, theta) + sigma2_tag * np.eye(self.num_tag) + self.jitter*np.eye(self.num_tag)

        L_ss = np.linalg.cholesky(K_ss)
        tmp1 = chol_inv(L_ss, self.src_y.T)
        tmp2 = chol_inv(L_ss, K_st)
        mu_t = np.dot(K_ts, tmp1)
        C_t  = K_tt - np.dot(K_ts, tmp2)

        L_t = np.linalg.cholesky(C_t)
        logDetCt = np.sum(np.log(np.diag(L_t)))
        delta = self.tag_y.T - mu_t
        alpha = chol_inv(L_t, delta)
        nlz = 0.5*(np.dot(delta.T, alpha) + self.num_tag*np.log(2*np.pi)) + logDetCt
        if(np.isnan(nlz)):
            nlz = np.inf

        self.nlz = nlz
        return nlz

    # Minimize the negative log-likelihood
    def train(self):
        theta0 = self.get_default_theta()
        print('theta before training', theta0)
        self.loss = np.inf
        self.theta = np.copy(theta0)
        hyp_bounds = [[None, None]] * (self.dim+3)
        hyp_bounds.extend([[-1,1]])

        nlz = self.neg_log_likelihood(theta0)
        print('nlz before training', nlz)

        def loss(theta):
            nlz = self.neg_log_likelihood(theta)
            return nlz

        def callback(theta):
            if self.nlz < self.loss:
                self.loss = self.nlz
                self.theta = np.copy(theta)

        gloss = value_and_grad(loss)

        try:
            fmin_l_bfgs_b(gloss, theta0, bounds=hyp_bounds, maxiter=self.bfgs_iter, m = 100, iprint=self.debug, callback=callback)
        except np.linalg.LinAlgError:
            print('TGP. Increase noise term and re-optimization')
            theta0 = np.copy(self.theta)
            theta0[self.dim+1] += np.log(10)
            theta0[self.dim+2] += np.log(10)
            try:
                fmin_l_bfgs_b(gloss, theta0, bounds=hyp_bounds, maxiter=self.bfgs_iter, m=10, iprint=self.debug, callback=callback)
            except:
                print('TGP. Exception caught, L-BFGS early stopping...')
                if self.debug:
                    print(traceback.format_exc())
        except:
            print('TGP. Exception caught, L-BFGS early stopping...')
            if self.debug:
                print(traceback.format_exc())

        if(np.isinf(self.loss) or np.isnan(self.loss)):
            print('TGP. Failed to build TGP model')
            sys.exit(1)

        print('TGP. TGP model training process finished')
        print('nlz after training', self.nlz)
        print('theta after training', self.theta)

    def predict(self, test_x, is_diag=1):
        output_scale = np.exp(self.theta[0])
        sigma2_tag = np.exp(self.theta[self.dim+2])
        C = self.kernel(self.src_x, self.tag_x, self.theta)
        L_C = np.linalg.cholesky(C)
        alpha_C = chol_inv(L_C, self.train_y.T)
        k_star_s = self.kernel2(test_x, self.src_x, self.theta)
        k_star_t = self.kernel1(test_x, self.tag_x, self.theta)
        k_star = np.hstack((k_star_s, k_star_t))
        py = np.dot(k_star, alpha_C)

        Cvks = chol_inv(L_C, k_star.T)
        if is_diag:
            ps2 = output_scale + sigma2_tag - (k_star * Cvks.T).sum(axis=1)
        else:
            ps2 = self.kernel1(test_x, test_x, self.theta) + sigma2_tag - np.dot(k_star, Cvks)
        ps2 = np.abs(ps2)
        py = py * self.std + self.mean
        ps2 = ps2 * (self.std**2)
        return py, ps2
   

