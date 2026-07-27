import torch 
import numpy as np
import matplotlib.pyplot as plt
import ray
import os

np.random.seed(0)

class DeterministicActor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=None):
        super(DeterministicActor, self).__init__()

        if hidden_size is None:
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(state_dim, action_dim),
            )
        else:
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(state_dim, hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_size, action_dim),
                #torch.nn.Sigmoid()  #map the output in [0,1]

                # decide if you want to append
                # another activation function
                # to map the output in a specific range
                # torch.nn.Tanh()
            )

    def forward(self, state):
        out = self.linear(state)
        return out

class EvolutionStrategies:
    def __init__(self, state_dim, action_dim,
                 hidden_size=None, env=None, 
                 batch_size=30, iter=100):
        self.actor = DeterministicActor(state_dim, action_dim, 
                                        hidden_size=hidden_size)
        self.mu = self.actor.state_dict().copy()
        self.std = self.actor.state_dict().copy()
        for param in self.mu.keys():
            self.mu[param] = torch.zeros_like(self.mu[param])
            self.std[param] = torch.ones_like(self.std[param])
        self.popsize = 100
        self.nbest = 5
        self.env = env
        self.batch_size = batch_size
        self.iter = iter

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state)
        return action.detach().numpy()[0]

    def set_weights(self, weights):
        self.actor.load_state_dict(weights)

    def get_weights(self):
        return self.actor.state_dict()

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename)

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename))

    def set_env(self, env):
        self.env = env

    def update(self):
        pop = []
        objs = np.empty(self.popsize)
        for p in range(self.popsize):
            sample = self.mu.copy()
            for param in self.mu.keys():
                sample[param] = torch.randn_like(sample[param]) \
                                    * self.std[param] \
                                    + self.mu[param]
            self.actor.load_state_dict(sample)
            simobjs = np.empty(self.batch_size)
            #np.random.seed(0)
            for b in range(self.batch_size):
                simobjs[b] = self.env.simulate()
            # plt.figure()
            # plt.boxplot(simobjs)
            # plt.show()
            objs[p] = np.mean(simobjs)
            pop.append(sample)
        sorted_indices = np.argsort(objs)
        objs = objs[sorted_indices]
        pop = [pop[i] for i in sorted_indices]
        pop = pop[:self.nbest]
        objs = np.mean(objs[:self.nbest])

        oldmu = self.mu.copy()
        for param in oldmu.keys():
            self.mu[param] = torch.stack([x[param] for x in pop]).mean(0)
            self.std[param] = torch.stack([x[param] for x in pop]).std(0)
        return objs

    def train(self, iter=None, batch_size=None,
              popsize=None, nbest=None):
        if iter is not None:
            self.iter = iter
        if batch_size is not None:
            self.batch_size = batch_size
        if popsize is not None:
            self.popsize = popsize
        if nbest is not None:
            self.nbest = nbest

        objs = np.empty(self.iter)
        for i in range(self.iter):
            objs[i] = self.update()
            print('Iteration :', i, 'Best :', objs[i])
        fig, ax = plt.subplots()
        plt.plot(objs)
        plt.xlabel('Iteration')
        plt.ylabel('Objective')
        # plt.show()
        self.actor.load_state_dict(self.mu)        



# Separable natural evolution strategies
class SNES:
    def __init__(self, state_dim, action_dim,
                 hidden_size=None, env=None, 
                 batch_size=30, iter=100,
                 popsize=10, dist=False):
        self.actor = DeterministicActor(state_dim, action_dim, 
                                        hidden_size=hidden_size)
        self.env = env
        self.batch_size = batch_size
        self.iter = iter
        self.popsize = popsize
        self.dist = dist

        # initialize mu and sigma 
        self.mu = torch.zeros(sum([torch.numel(l) for l in self.actor.parameters()]))
        self.sigma = torch.ones(sum([torch.numel(l) for l in self.actor.parameters()]))

        # set default learning rates
        self.mu_lr = 1.0 * 0.5
        self.sigma_lr = 0.5* (0.2 * (3 + np.log(self.mu.shape[0])) / np.sqrt(self.mu.shape[0]))


    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state)
        return action.detach().numpy()[0]

    def set_weights(self, weights):
        self.actor.load_state_dict(weights)

    def get_weights(self):
        return self.actor.state_dict()
    
    def save(self, filename):
        torch.save(self.actor.state_dict(), filename)
    
    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename))

    
    def train(self, label, iter=None, batch_size=None,
              popsize=None, dist=None, agg = 'median', percentile = 70):
    
        if iter is not None:
            self.iter = iter
        if batch_size is not None:
            self.batch_size = batch_size
        if popsize is not None:
            self.popsize = popsize
        if dist is not None:
            self.dist = dist
        
        # create actors if distributed
        if self.dist:
            ray.init(num_cpus=self.popsize)
            self.simulators = [Simulator.remote(self.env.copy()) \
                               for _ in range(self.popsize)]
            #self.simulators = [Simulator.remote(self.env) \
                            #for _ in range(self.popsize)]

        objs = np.empty(self.iter)
        for i in range(self.iter):
            objs[i] = self.step()
            #print('Iteration :', i, 'Best :', objs[i])
            if all((torch.abs(self.mu) + self.sigma)/torch.abs(self.mu) < 1.01):
                break
        fig, ax = plt.subplots()
        plt.plot(np.arange(40, len(objs)), objs[40:])
        plt.xlabel('Iteration')
        plt.ylabel('Objective')
        save_dir = f'results/figures/{label}'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/objective_vs_iteration.png')
        plt.close(fig)
        #plt.show()
        # 保存objs数组
        np.save(f'{save_dir}/objective_vs_iteration.npy', objs)
        
        with torch.no_grad(): #关闭梯度计算，只是赋值，不需要反向传播
            # set parameters of policy
            idx = 0
            new_pol = self.actor.state_dict().copy()
            for param in new_pol.keys():
                new_pol[param] = self.mu[idx:idx+torch.numel(new_pol[param])].reshape(new_pol[param].shape)
                idx += torch.numel(new_pol[param])
                # 从一维向量 self.mu 中，取出当前参数需要的元素（比如4x3=12个），用 .reshape(new_pol[param].shape) 还原成原来参数的形状（比如[4,3]），赋值给 new_pol[param]

            self.actor.load_state_dict(new_pol)
            # SNES 训练时把所有参数“拉平”为一个大向量（方便采样和更新）。
             #但PyTorch神经网络需要的是“分块”的参数结构（每层一个张量）。
             #所以采样/优化后，需要把一维向量拆分还原成各层参数，再加载到网络里。

    def step(self, agg = 'median', percentile = 70):
        # sample noise
        noise = torch.randn(self.popsize, *self.mu.shape)
        #self.mu 是所有参数拉平后的一维均值向量

        sampled_pop = self.mu + self.sigma * noise  # sample popsize policy parmas, from N(mu, sigma^2)

        if self.dist == False:
            objs = np.empty(self.popsize)
            for p in range(self.popsize):
            
                with torch.no_grad():
                    # set parameters of policy
                    idx = 0
                    new_pol = self.actor.state_dict().copy()
                    for param in new_pol.keys():  #把一位参数向量包还原成网络参数结构
                        new_pol[param] = sampled_pop[p][idx:idx+torch.numel(new_pol[param])].reshape(new_pol[param].shape)
                        idx += torch.numel(new_pol[param])

                    self.actor.load_state_dict(new_pol)
                    simobjs = np.empty(self.batch_size)
                    #np.random.seed(0)
                    for b in range(self.batch_size): # 仿真batch——size次
                        simobjs[b] = self.env.simulate()
                    # plt.figure()
                    # plt.boxplot(simobjs)
                    # plt.show()
                    objs[p] = np.mean(simobjs)
        else: #分布式仿真
            futures = [self.simulators[p].simulate_policy\
                       .remote(sampled_pop[p], self.batch_size) \
                                for p in range(self.popsize)]
            objs = np.array(ray.get(futures)).flatten()

        objs = torch.from_numpy(objs)

        # sort objectives and transform into fitness using rank transformation
        sorted_indices = torch.argsort(objs)
        rank = torch.argsort(sorted_indices)
        fitness = 1 + rank
        #用秩变化把表现换成fitness
        fitness = torch.max(torch.tensor(0), torch.log(torch.tensor(self.popsize/2 + 1)) - torch.log(fitness))
        fitness = fitness / torch.sum(fitness) - 1 / self.popsize

        # compute gradients
        grad_mu = torch.sum(fitness.unsqueeze(-1) * noise, 0)
        grad_sigma = torch.sum(fitness.unsqueeze(-1) * (noise**2 - 1), 0)

        # update mu and sigma
        self.mu += self.mu_lr * self.sigma * grad_mu
        self.sigma *= torch.exp(self.sigma_lr / torch.tensor(2) * grad_sigma)

        #print(self.mu)
        #print(self.sigma)

        #return torch.median(objs).item()

        # if agg == 'median':
        #     return torch.median(objs).item()
        # elif agg == 'mean':
        #     return torch.mean(objs).item()
        # elif agg == 'percentile':
        #     return np.percentile(objs.numpy(), percentile)
        # else:
        #     raise ValueError("agg must be 'median', 'mean', or 'percentile'")

        return np.percentile(objs.numpy(),70)



# ray actor (class) for distributed simulation 
@ray.remote
class Simulator:
    # initialize with a copy of the environment for simulation
    def __init__(self, env):
        self.env = env.copy()

    # simulate the policy
    def simulate_policy(self, policy, batch_size, agg = 'mean',percentile = 70):

        with torch.no_grad():
            # set parameters of policy
            idx = 0
            new_pol = self.env.policy.actor.state_dict().copy()
            for param in new_pol.keys():
                new_pol[param] = policy[idx:idx+torch.numel(new_pol[param])].reshape(new_pol[param].shape)
                idx += torch.numel(new_pol[param])

            self.env.policy.actor.load_state_dict(new_pol)
            simobjs = np.empty(batch_size)
            # np.random.seed(0)
            for b in range(batch_size):
                simobjs[b] = self.env.simulate()
        
        if agg == 'mean':
            return np.mean(simobjs)   
        elif agg == 'median':
            return np.median(simobjs)
        elif agg == 'percentile':
            return np.percentile(simobjs, percentile)
        else:
            raise ValueError("agg must be 'mean', 'median', or 'percentile'")