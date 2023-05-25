import numpy as np
import argparse
import os

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--activation')
parser.add_argument('--fmap')
parser.add_argument('--i')
parser.add_argument('--dataset')
args = parser.parse_args()


save_dir = os.path.join(args.dataset, args.fmap, args.activation)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


if args.activation == 'square':
    phi = square
    dev_phi = dev_square
    dev2_phi = dev2_square
elif args.activation == 'tanh':
    phi = tanh
    dev_phi = dev_tanh
    dev2_phi = dev2_tanh
elif args.activation == 'cos':
    phi = cos
    dev_phi = dev_cos
    dev2_phi = dev2_cos
elif args.activation == 'softplus':
    phi = softplus
    dev_phi = dev_softplus
    dev2_phi = dev2_softplus


Ns = [400, 800, 1200]


if args.dataset == 'synthetic':
    d = 1000
elif args.dataset == 'MNIST':
    d = 784
elif args.dataset == 'CIFAR-10':
    d = 3072


if 'rf' in args.fmap:
    ks = [i * 10000 for i in range(1, 21)]
elif 'ntk' in args.fmap:
    ks = [i * 15 for i in range(1, 21)]



for N in Ns:
    for k in ks:
        
        if args.dataset == 'synthetic':
            X = np.random.randn(N, d)
            Y = np.random.randn(N)
            z = np.random.randn(d)
        else:
            class1 = 0
            class2 = 1
            X, z, Y, _ = load_dataset(args.dataset, N, class1, class2)

        W = np.random.randn(d, k) / np.sqrt(d)

        if 'rf' in args.fmap:
            Z = phi(X @ W)
            nabla_phi_z = W @ np.diag(dev_phi(W.transpose() @ z))
        elif 'ntk' in args.fmap:
            Z = rwt(X, dev_phi(X @ W))
            nabla_phi_z = rwt(np.identity(d), np.outer(np.ones(d), dev_phi(W.transpose() @ z))) + \
            rwt(np.outer(np.ones(d), z), W @ np.diag(dev2_phi(W.transpose() @ z)))

        S = np.sqrt(d) * np.linalg.norm(nabla_phi_z @ np.linalg.pinv(Z) @ Y)

        with open(os.path.join(save_dir, args.i + '.txt'), 'a') as f:
            f.write(str(d) + '\t' + str(k) + '\t' + str(N) + '\t' + str(S) + '\n')
