import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time
import warnings


class LowRankMixtureModel(torch.nn.Module):
    """
    Initialize a probabilistic model representing either a Mixture of Factor 
    Analyzers (MFA) [1] or a Mixture of Probabilistic Principal Component 
    Analyzers (MPPCA) [2]. These models constrain the covariance matrices of a 
    Gaussian mixture model to be low-rank and diagonal. For MFA, the noise 
    matrix is assumed to be diagonal; MPPCA assumes that the noise matrix is
    both diagonal and isotropic.
    
    Original publications:
    [1] Tipping, M. E., & Bishop, C. M. (1999). Mixtures of Probabilistic 
        Principal Component Analyzers. Neural Computation, 11(2), 443-482.

    [2] Ghahramani, Z., & Hinton, G. E. (1996). The EM Algorithm for Mixtures of
        Factor Analyzers (Vol. 60). Technical Report CRG-TR-96-1, University of 
        Toronto.
        
    The implementation is based on the open-source code from
    [3] Richardson, E., & Weiss, Y. (2018). On GANs and GMMs. Advances in 
        Neural Information Processing Systems, 31.

    Parameters:
        waypoints (list of np.array): list of waypoints as numpy arrays
        stop_duration (float): duration of pallet pause at each waypoint (seconds)
        max_speed (float): maximum speed of the pallet (meters per second)
        acc (float): acceleration of the pallet (meters per second^2)
        dt (float): simulation timestep (seconds)

    """
    def __init__(self, n_components, n_features, n_factors, isotropic_noise=True, init_method='rnd_samples'):
        super(LowRankMixtureModel, self).__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.n_factors = n_factors
        self.init_method = init_method
        self.isotropic_noise = isotropic_noise

        self.mu = torch.nn.Parameter(torch.zeros(n_components, n_features), requires_grad=False)
        self.W = torch.nn.Parameter(torch.zeros(n_components, n_features, n_factors), requires_grad=False)
        self.log_Psi = torch.nn.Parameter(torch.zeros(n_components, n_features), requires_grad=False)
        self.pi_logits = torch.nn.Parameter(torch.log(torch.ones(n_components)/float(n_components)), requires_grad=False)


    def sample(self, n, with_noise=False):
        K, d, l = self.W.shape
        sampled_components = torch.multinomial(torch.softmax(self.pi_logits, dim=0), n, replacement=True)
        z_l = torch.randn(n, l, device=self.W.device)

        if with_noise:
            z_d = torch.randn(n, d, device=self.W.device)  
        else:
            z_d = torch.zeros(n, d, device=self.W.device)
        
        Wz = self.W[sampled_components] @ z_l[..., None]
        mu = self.mu[sampled_components][..., None]
        epsilon = (z_d * torch.exp(0.5*self.log_Psi[sampled_components]))[..., None]
        
        samples = Wz + mu + epsilon

        return samples.squeeze(), sampled_components
    

    def _component_log_likelihood(self, x, pi, mu, W, sigma2):
        K, d, l = W.shape
        WT = W.transpose(1,2)
        #inv_sigma2 = (1.0/sigma2 * torch.ones(d,1)).view(d,K,1).transpose(0,1)
        inv_sigma2 = torch.exp(-sigma2).view(K, d, 1)
        I = torch.eye(l, device=W.device).reshape(1,l,l)
        L = I + WT @ (inv_sigma2 * W)
        inv_L = torch.linalg.solve(L, I)

        # compute Mahalanobis distance using the matrix inversion lemma
        def mahalanobis_distance(i):
            x_c = (x - mu[i].reshape(1,d)).T
            component_m_d = (inv_sigma2[i] * x_c) - \
                ((inv_sigma2[i] * W[i]) @ inv_L[i]) @ (WT[i] @ (inv_sigma2[i] * x_c))
            
            return torch.sum(x_c * component_m_d, dim=0)

        # combine likelihood terms
        m_d = torch.stack([mahalanobis_distance(i) for i in range(K)])
        log_det_cov = torch.logdet(L) - \
            torch.sum(torch.log(inv_sigma2.reshape(K, d)), axis=1)
        log_const = torch.log(torch.tensor(2.0)*torch.pi)
        log_probs = -0.5 * ((d*log_const + log_det_cov).reshape(K, 1) + m_d)

        return torch.log(pi).reshape(1, K) + log_probs.T



    def per_component_log_likelihood(self, x, sampled_features=None):
        if sampled_features is not None:
            return self._component_log_likelihood(x[:, sampled_features], torch.softmax(self.pi_logits, dim=0),
                                                 self.mu[:, sampled_features],
                                                 self.W[:, sampled_features],
                                                 self.log_Psi[:, sampled_features])
        return self._component_log_likelihood(x, torch.softmax(self.pi_logits, dim=0), self.mu, self.W, self.log_Psi)


    def log_prob(self, x, sampled_features=None):
        return torch.logsumexp(self.per_component_log_likelihood(x, sampled_features), dim=1)


    def responsibilities(self, x, sampled_features=None):
        comp_LLs = self.per_component_log_likelihood(x, sampled_features)
        log_responsibilities = comp_LLs - self.log_prob(x, sampled_features).reshape(-1, 1)

        return torch.exp(log_responsibilities)


    def _small_sample_ppca(self, x, n_factors):
        # See https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
        mu = torch.mean(x, dim=0)
        U, S, V = torch.svd(x - mu.reshape(1, -1))

        V = V.T.to(x.device)
        S = S.to(x.device)
        # (3.13) in Tipping and Bishop (1999) [1]
        sigma2 = torch.sum(S[n_factors:]**2.0)/((x.shape[0]-1) * (x.shape[1]-n_factors))
        # (3.12) in Tipping and Bishop (1999) [1]
        W = V[:, :n_factors] * torch.sqrt((S[:n_factors]**2.0).reshape(1, n_factors)/(x.shape[0]-1) - sigma2)

        return mu, W, torch.log(sigma2) * torch.ones(x.shape[1], device=x.device)
    
    
    def _init_from_data(self, x, samples_per_component, feature_sampling=False):
        n = x.shape[0]
        K, d, l = self.W.shape

        if self.init_method == 'kmeans':
            # Import this only if 'kmeans' method was selected (not sure this is a good practice...)
            from sklearn.cluster import KMeans
            sampled_features = np.random.choice(d, int(d*feature_sampling)) if feature_sampling else np.arange(d)

            t = time.time()
            print('Performing K-means clustering of {} samples in dimension {} to {} clusters...'.format(
                x.shape[0], sampled_features.size, K))
            _x = x[:, sampled_features].cpu().numpy()
            clusters = KMeans(n_clusters=K, max_iter=300).fit(_x)
            print('... took {} sec'.format(time.time() - t))
            component_samples = [clusters.labels_ == i for i in range(K)]
        elif self.init_method == 'rnd_samples':
            m = samples_per_component
            o = np.random.choice(n, m*K, replace=False) if m*K < n else np.arange(n)
            assert n >= m*K
            component_samples = [[o[i*m:(i+1)*m]] for i in range(K)]

        params = [torch.stack(t) for t in zip(
            *[self._small_sample_ppca(x[component_samples[i]], n_factors=l) for i in range(K)])]

        self.mu.data = params[0]
        self.W.data = params[1]
        self.log_Psi.data = params[2]


    def fit(self, x, max_iterations=20, feature_sampling=False):
        """
        Estimate Maximum Likelihood MPPCA parameters for the provided data using EM per
        Tipping, and Bishop. Mixtures of probabilistic principal component analyzers.
        :param x: training data (arranged in rows), shape = (<numbr of samples>, n_features)
        :param max_iterations: number of iterations
        :param feature_sampling: allows faster responsibility calculation by sampling data coordinates
        """
        assert self.isotropic_noise, 'EM fitting is currently supported for isotropic noise (MPPCA) only'
        assert not feature_sampling or type(feature_sampling) == float, 'set to desired sampling ratio'
        K, d, l = self.W.shape
        N = x.shape[0]

        print('Random init...')
        init_samples_per_component = (l+1)*2 if self.init_method == 'rnd_samples' else (l+1)*10
        self._init_from_data(x, samples_per_component=init_samples_per_component,
                             feature_sampling=feature_sampling/2 if feature_sampling else False)
        print('Init log-likelihood ={:.4f}'.format(torch.mean(self.log_prob(x)).item()))

        def per_component_m_step(i):
            mu_i = torch.sum(r[:, [i]] * x, dim=0) / r_sum[i]
            s2_I = torch.exp(self.log_Psi[i, 0]) * torch.eye(l, device=x.device)
            inv_M_i = torch.inverse(self.W[i].T @ self.W[i] + s2_I)
            x_c = x - mu_i.reshape(1, d)
            SiAi = (1.0/r_sum[i]) * (r[:, [i]]*x_c).T @ (x_c @ self.W[i])
            invM_AT_Si_Ai = inv_M_i @ self.W[i].T @ SiAi
            A_i_new = SiAi @ torch.inverse(s2_I + invM_AT_Si_Ai)
            t1 = torch.trace(A_i_new.T @ (SiAi @ inv_M_i))   # (eq) 6 in [2]
            trace_S_i = torch.sum(N/r_sum[i] * torch.mean(r[:, [i]]*x_c*x_c, dim=0)) # (eq) 6 in [2]
            sigma_2_new = (trace_S_i - t1)/d # (eq) 6 in [2]
            return mu_i, A_i_new, torch.log(sigma_2_new) * torch.ones_like(self.log_Psi[i])


        ll_log = []
        for it in range(max_iterations):
            t = time.time()
            sampled_features = np.random.choice(d, int(d*feature_sampling)) if feature_sampling else None
            r = self.responsibilities(x, sampled_features=sampled_features)
            r_sum = torch.sum(r, dim=0)
            new_params = [torch.stack(t) for t in zip(*[per_component_m_step(i) for i in range(K)])]
            self.mu.data = new_params[0]
            self.W.data = new_params[1]
            self.log_Psi.data = new_params[2]
            self.pi_logits.data = torch.log(r_sum / torch.sum(r_sum))
            ll = torch.mean(self.log_prob(x)).item() #if it % 5 == 0 else '.....'
            print('Iteration {}/{}, train log-likelihood = {:.4f}, took {:.4f} sec'.format(it, max_iterations, ll,
                                                                                   time.time()-t))
            ll_log.append(ll)
        return ll_log

    def batch_fit(self, train_dataset, test_dataset=None, batch_size=1000, test_size=1000, max_iterations=20,
                  feature_sampling=False):
        """
        Estimate Maximum Likelihood MPPCA parameters for the provided data using EM per
        Tipping, and Bishop. Mixtures of probabilistic principal component analyzers.
        Memory-efficient batched implementation for large datasets that do not fit in memory:
        E step:
            For all mini-batches:
            - Calculate and store responsibilities
            - Accumulate sufficient statistics
        M step: 
            Re-calculate all parameters
        Note that incremental EM per Neal & Hinton, 1998 is not supported, since we can't maintain
            the full x x^T as sufficient statistic - we need to multiply by A to get a more compact
            representation.
        :param train_dataset: pytorch Dataset object containing the training data (will be iterated over)
        :param test_dataset: optional pytorch Dataset object containing the test data (otherwise train_daset will be used)
        :param batch_size: the batch size
        :param test_size: number of samples to use when reporting likelihood
        :param max_iterations: number of iterations (=epochs)
        :param feature_sampling: allows faster responsibility calculation by sampling data coordinates
       """
        assert self.isotropic_noise, 'EM fitting is currently supported for isotropic noise (MPPCA) only'
        assert not feature_sampling or type(feature_sampling) == float, 'set to desired sampling ratio'
        K, d, l = self.W.shape

        init_samples_per_component = (l+1)*2 if self.init_method == 'rnd_samples' else (l+1)*10
        print('Random init using {} with {} samples per component...'.format(self.init_method, init_samples_per_component))
        init_keys = [key for i, key in enumerate(RandomSampler(train_dataset)) if i < init_samples_per_component*K]
        init_samples, _ = zip(*[train_dataset[key] for key in init_keys])
        self._init_from_data(torch.stack(init_samples).to(self.mu.device),
                             samples_per_component=init_samples_per_component,
                             feature_sampling=feature_sampling/2 if feature_sampling else False)

        # Read some test samples for test likelihood calculation
        # test_samples, _ = zip(*[test_dataset[key] for key in RandomSampler(test_dataset, num_samples=test_size, replacement=True)])
        test_dataset = test_dataset or train_dataset
        all_test_keys = [key for key in SequentialSampler(test_dataset)]
        test_samples, _ = zip(*[test_dataset[key] for key in all_test_keys[:test_size]])
        test_samples = torch.stack(test_samples).to(self.mu.device)

        ll_log = []
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        for it in range(max_iterations):
            t = time.time()

            # Sufficient statistics
            sum_r = torch.zeros(size=[K], dtype=torch.float64, device=self.mu.device)
            sum_r_x = torch.zeros(size=[K, d], dtype=torch.float64, device=self.mu.device)
            sum_r_x_x_A = torch.zeros(size=[K, d, l], dtype=torch.float64, device=self.mu.device)
            sum_r_norm_x = torch.zeros(K, dtype=torch.float64, device=self.mu.device)

            ll_log.append(torch.mean(self.log_prob(test_samples)).item())
            print('Iteration {}/{}, log-likelihood={}:'.format(it, max_iterations, ll_log[-1]))

            for batch_x, _ in loader:
                print('E', end='', flush=True)
                batch_x = batch_x.to(self.mu.device)
                sampled_features = np.random.choice(d, int(d*feature_sampling)) if feature_sampling else None
                batch_r = self.responsibilities(batch_x, sampled_features=sampled_features)
                sum_r += torch.sum(batch_r, dim=0).double()
                sum_r_norm_x += torch.sum(batch_r * torch.sum(torch.pow(batch_x, 2.0), dim=1, keepdim=True), dim=0).double()
                for i in range(K):
                    batch_r_x = batch_r[:, [i]] * batch_x
                    sum_r_x[i] += torch.sum(batch_r_x, dim=0).double()
                    sum_r_x_x_A[i] += (batch_r_x.T @ (batch_x @ self.W[i])).double()

            print(' / M...', end='', flush=True)
            self.pi_logits.data = torch.log(sum_r / torch.sum(sum_r)).float()
            self.mu.data = (sum_r_x / sum_r.reshape(-1, 1)).float()
            SA = sum_r_x_x_A / sum_r.reshape(-1, 1, 1) - \
                 (self.mu.reshape(K, d, 1) @ (self.mu.reshape(K, 1, d) @ self.W)).double()
            s2_I = torch.exp(self.log_Psi[:, 0]).reshape(K, 1, 1) * torch.eye(l, device=self.mu.device).reshape(1, l, l)
            M = (self.W.transpose(1, 2) @ self.W + s2_I).double()
            inv_M = torch.stack([torch.inverse(M[i]) for i in range(K)])   # (K, l, l)
            invM_AT_S_A = inv_M @ self.W.double().transpose(1, 2) @ SA   # (K, l, l)
            self.W.data = torch.stack([(SA[i] @ torch.inverse(s2_I[i].double() + invM_AT_S_A[i])).float()
                                       for i in range(K)])
            t1 = torch.stack([torch.trace(self.W[i].double().T @ (SA[i] @ inv_M[i])) for i in range(K)])
            t_s = sum_r_norm_x / sum_r - torch.sum(torch.pow(self.mu, 2.0), dim=1).double()
            self.log_Psi.data = torch.log((t_s - t1)/d).float().reshape(-1, 1) * torch.ones_like(self.log_Psi)

            #self._parameters_sanity_check()
            print(' ({} sec)'.format(time.time()-t))

        ll_log.append(torch.mean(self.log_prob(test_samples)).item())
        print('\nFinal train log-likelihood={}:'.format(ll_log[-1]))
        return ll_log

    def sgd_mfa_train(self, train_dataset, test_dataset=None, batch_size=128, test_size=1000, max_epochs=10,
                      learning_rate=0.001, feature_sampling=False):
        """
        Stochastic Gradient Descent training of MFA (after initialization using MPPCA EM)
        :param train_dataset: pytorch Dataset object containing the training data (will be iterated over)
        :param test_dataset: optional pytorch Dataset object containing the test data (otherwise train_daset will be used)
        :param batch_size: the batch size
        :param test_size: number of samples to use when reporting likelihood
        :param max_epochs: number of epochs
        :param feature_sampling: allows faster responsibility calculation by sampling data coordinates
        """
        if torch.all(self.W == 0.):
            warnings.warn('SGD MFA training requires initialization. Please call batch_fit() first.')
        if self.isotropic_noise:
            warnings.warn('Currently, SGD training uses diagonal (non-isotropic) noise covariance i.e. MFA and not MPPCA')
        assert not feature_sampling or type(feature_sampling) == float, 'set to desired sampling ratio'
        # self.pi_logits.requires_grad =
        self.mu.requires_grad = self.W.requires_grad = self.log_Psi.requires_grad = True
        K, d, l = self.W.shape

        # Read some test samples for test likelihood calculation
        # test_samples, _ = zip(*[test_dataset[key] for key in RandomSampler(test_dataset, num_samples=test_size, replacement=True)])
        test_dataset = test_dataset or train_dataset
        all_test_keys = [key for key in SequentialSampler(test_dataset)]
        test_samples, _ = zip(*[test_dataset[key] for key in all_test_keys[:test_size]])
        test_samples = torch.stack(test_samples).to(self.mu.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        ll_log = []
        self.train()
        for epoch in range(max_epochs):
            t = time.time()
            for idx, (batch_x, _) in enumerate(loader):
                #print('.', end='', flush=True)
                if idx > 0 and idx%100 == 0:
                    print("Iteration {}".format(idx))
                    print("Loss {:.8f}".format(torch.mean(self.log_prob(test_samples)).item()))
                    #print(torch.mean(self.log_prob(test_samples)).item())
                sampled_features = np.random.choice(d, int(d*feature_sampling)) if feature_sampling else None
                batch_x = batch_x.to(self.mu.device)
                optimizer.zero_grad()
                loss = -torch.sum(self.log_prob(batch_x, sampled_features=sampled_features)) / batch_size
                #loss = -torch.mean(self.log_prob(test_samples, sampled_features=sampled_features))
                loss.backward()
                optimizer.step()
            ll_log.append(torch.mean(self.log_prob(test_samples)).item())
            print('\nEpoch {}: Test ll = {:.4f} ({:.4f} sec)'.format(epoch, ll_log[-1], time.time()-t))
            #self._parameters_sanity_check()
        self.pi_logits.requires_grad = self.mu.requires_grad = self.W.requires_grad = self.log_Psi.requires_grad = False
        return ll_log