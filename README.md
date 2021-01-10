# Transfer-GP

## About
An adaptive transfer learning algorithm based on Gaussian Process, which can be used to adapt the transfer learning schemes by automatically estimating the similarity between a source and a target task.

The regression results comparisons between conventional Gaussian process method and transfer Gaussian process method are shown here.

![image](https://github.com/Xiao-dong-Wang/Transfer-GP/blob/master/figures/test1_GP.png)

![image](https://github.com/Xiao-dong-Wang/Transfer-GP/blob/master/figures/test1_TGP.png)

Codes reimplemented here is based on the idea from the following paper:

- Bin Cao, Sinno Jialin Pan, Yu Zhang, Dit-Yan Yeung, Qiang Yang, Adaptive transfer learning, *Twenty-Fourth Conference on Artificial Intelligence*
(AAAI), 2010.

Dependencies:

Autograd: https://github.com/HIPS/autograd

Scipy: https://github.com/scipy/scipy
