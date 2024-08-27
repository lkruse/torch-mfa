import numpy as np
from sklearn.cluster import KMeans
import time
import torch
from torch.utils.data import DataLoader, RandomSampler


class LowRankMixtureModel(torch.nn.Module):
    """
    Initialize a probabilistic model representing either a Mixture of Factor 
    Analyzers (MFA) [1] or a Mixture of Probabilistic Principal Component 
    Analyzers (MPPCA) [2]. These models constrain the covariance matrices of a 
    Gaussian mixture model to be low-rank and diagonal. For MFA, the noise 
    matrix is assumed to be diagonal; MPPCA assumes that the noise matrix is
    both diagonal and isotropic.
    
    Original publications:
    [1] Ghahramani, Z., & Hinton, G. E. (1996). The EM Algorithm for Mixtures of
        Factor Analyzers (Vol. 60). Technical Report CRG-TR-96-1, University of 
        Toronto.

    [2] Tipping, M. E., & Bishop, C. M. (1999). Mixtures of Probabilistic 
        Principal Component Analyzers. Neural Computation, 11(2), 443-482.

    The implementation is based on the open-source code from
    [3] Richardson, E., & Weiss, Y. (2018). On GANs and GMMs. Advances in 
        Neural Information Processing Systems, 31.
        
    Problem Parameters:
        n_components (int): number of mixture components (alias: K)
        n_features (int): number of input dimensions (alias: d)
        n_factors (int): number of underlying factors (alias: l)
        init_method: (str): initialization method: 'kmeans' or 'rnd_samples'
    
    Learnable Parameters:
        mu (torch.Tensor): [K x d] tensor of component mean vectors
        W (torch.Tensor): [K x d x l] tensor of factor loading matrices
        log_Psi (torch.Tensor): [K x d] tensor of log diagonal noise values
        pi_logits (torch.Tensor): [K] tensor of mixing proportion logits

    """
    def __init__(self, n_components, n_features, n_factors, init_method='rnd_samples'):
        super(LowRankMixtureModel, self).__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.n_factors = n_factors
        self.init_method = init_method

        self.mu = torch.nn.Parameter(torch.zeros(n_components, n_features), requires_grad=False)
        self.W = torch.nn.Parameter(torch.zeros(n_components, n_features, n_factors), requires_grad=False)
        self.log_Psi = torch.nn.Parameter(torch.zeros(n_components, n_features), requires_grad=False)
        self.pi_logits = torch.nn.Parameter(torch.log(torch.ones(n_components)/float(n_components)), requires_grad=False)


    def sample(self, n, with_noise=False):
        """
        Sample from the learned generative model.

        Parameters:
        n (int): total number of samples to draw
        with_noise (boolean): sample with the learned noise model

        Returns:
        samples (tensor): [n x d] tensor of generated samples
        sampled_compoments (torch.Tensor): [n] tensor containing the underlying
            component that generated each sample
        """
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
    

    def component_log_likelihood(self, x, mu, W, log_Psi, pi_logits):
        """
        Compute the log-likelihoods associated with each mixture component.

        Parameters:
        x (torch.Tensor): [n x d] tensor of input data
        mu (torch.Tensor): [K x d] tensor of component mean vectors
        W (torch.Tensor): [K x d x l] tensor of factor loading matrices
        log_Psi (torch.Tensor): [K x d] tensor of log diagonal noise values
        pi_logits (torch.Tensor): [K] tensor of mixing proportion logits 

        Returns:
        component_lls (torch.Tensor): size [n x K] tensor of per-component log-likelihoods 
        """
        K, d, l = W.shape
        WT = W.transpose(1,2)
        inv_Psi = torch.exp(-log_Psi).view(K, d, 1)
        I = torch.eye(l, device=W.device).reshape(1,l,l)
        L = I + WT @ (inv_Psi * W)
        inv_L = torch.linalg.solve(L, I)

        # compute Mahalanobis distance using the matrix inversion lemma
        def mahalanobis_distance(i):
            x_c = (x - mu[i].reshape(1,d)).T
            component_m_d = (inv_Psi[i] * x_c) - \
                ((inv_Psi[i] * W[i]) @ inv_L[i]) @ (WT[i] @ (inv_Psi[i] * x_c))
            
            return torch.sum(x_c * component_m_d, dim=0)

        # combine likelihood terms
        m_d = torch.stack([mahalanobis_distance(i) for i in range(K)])
        log_det_cov = torch.logdet(L) - \
            torch.sum(torch.log(inv_Psi.reshape(K, d)), axis=1)
        log_const = torch.log(torch.tensor(2.0)*torch.pi)
        log_probs = -0.5 * ((d*log_const + log_det_cov).reshape(K, 1) + m_d)

        return pi_logits.reshape(1, K) + log_probs.T


    def per_component_log_likelihood(self, x, sampled_features=None):
        """
        Computes the per-sample and per-component log-likelihoods with feature 
        sampling, if applicable.

        Parameters:
        x (torch.Tensor): [n x d] tensor of input data
        sampled_features (list): [K x d] list of feature coordinates to use

        Returns:
        lls (torch.Tensor): size [n x K] tensor of per-component log-likelihoods 
        """
        if sampled_features is not None:
            component_lls = self.component_log_likelihood(x[:, sampled_features], 
                                                  self.mu[:, sampled_features],
                                                  self.W[:, sampled_features],  
                                                  self.log_Psi[:, sampled_features],
                                                  self.pi_logits)
        else:
            component_lls = self.component_log_likelihood(x,self.mu, self.W, self.log_Psi, self.pi_logits)

        return component_lls
    

    def per_sample_log_likelihood(self, x, sampled_features=None):
        """
        Computes the per-sample log-likelihoods.

        Parameters:
        x (torch.Tensor): [n x d] tensor of input data
        sampled_features (list): [K x d] list of feature coordinates to use

        Returns:
        sample_lls (torch.Tensor): size [n] tensor of per-sample log-likelihoods 
        """
        sample_lls = torch.logsumexp(self.per_component_log_likelihood(x, sampled_features), dim=1)

        return sample_lls


    def responsibilities(self, x, sampled_features=None):
        """
        Compute the responsibility of each component for generating each sample.

        Parameters:
        x (torch.Tensor): [n x d] tensor of input data
        sampled_features (list): [K x d] list of feature coordinates to use

        Returns:
        responsibilities (torch.Tensor): size [n x K] tensor of component responsibilities
        """
        comp_LLs = self.per_component_log_likelihood(x, sampled_features)
        log_responsibilities = comp_LLs - self.per_sample_log_likelihood(x, sampled_features).reshape(-1, 1)
        responsibilities = torch.exp(log_responsibilities)
        
        return responsibilities


    def map_component(self, x, sampled_features=None):
        """
        Return the maximum a posteriori (MAP) component numbers.

        Parameters:
        x (torch.Tensor): [n x d] tensor of input data
        sampled_features (list): [K x d] list of feature coordinates to use

        Returns:
        map_components (torch.Tensor): size [n] tensor of MAP component assignments
        """
        map_components = torch.argmax(self.responsibilities(x, sampled_features), dim=1)

        return map_components


    def small_sample_ppca(self, x, n_factors):
        """
        Solve for the parameters of a single mixture component exactly using
        singular value decomposition given an initial clustering.

        Parameters:
        x (torch.Tensor): [ni x d] tensor of input data where ni is the
            number of data points assigned to cluster i
        n_factors (int): number of underlying factors (alias: l)

        Returns:
        mu (torch.Tensor): [d] mean vector
        W (torch.tensor): [d x l] factor loading matrix
        sigma2 (torch.Tensor): scalar noise variance
        """
        # See https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
        mu = torch.mean(x, dim=0)
        U, S, V = torch.linalg.svd(x - mu.reshape(1, -1), full_matrices=False)

        V = V.T.to(self.mu.device)
        S = S.to(self.mu.device)
        # All equations and appendices reference Tipping and Bishop (1999) [2]
        # (3.13)
        sigma2 = torch.sum(S[n_factors:]**2.0)/((x.shape[0]-1) * (x.shape[1]-n_factors))
        # (3.12)
        W = V[:, :n_factors] * torch.sqrt((S[:n_factors]**2.0).reshape(1, n_factors)/(x.shape[0]-1) - sigma2)

        return mu, W, torch.log(sigma2) * torch.ones(x.shape[1], device=self.mu.device)

    
    def init_from_data(self, x, samples_per_component, feature_sampling=False):
        """
        Initialize the parameter values by first assigning data points to 
        component clusters (using either K-Means or random assignment) and then
        solving for the mixture parameters using singular value decomposition.

        Parameters:
        x (torch.Tensor): [n x d] tensor of input data
        samples_per_component (int): the number of samples to assign to each 
            component when using the 'rnd_samples' method
        feature_sampling (float or False): if float, denotes the fraction of
            total features to sample to speed up responsibility calculation
        """
        n = x.shape[0]
        K, d, l = self.W.shape

        if self.init_method == 'kmeans':
            if feature_sampling:
                sampled_features = torch.multinomial(torch.ones(d), int(d*feature_sampling), replacement=False) 
            else:
                sampled_features = torch.arange(d)
            t = time.time()
            print('Performing K-means clustering of {} samples with {} dimensions to {} clusters...'.format(
                x.shape[0], sampled_features.shape[0], K))
            _x = x[:, sampled_features].cpu().numpy()
            clusters = KMeans(n_clusters=K, max_iter=300).fit(_x)
            print('... took {:.4f} sec'.format(time.time() - t))
            component_samples = [clusters.labels_ == i for i in range(K)]
        elif self.init_method == 'rnd_samples':
            m = samples_per_component
            o = torch.multinomial(torch.ones(n), m*K, replacement=False) if m*K < n else torch.arange(n)
            assert n >= m*K
            component_samples = [[o[i*m:(i+1)*m]] for i in range(K)]

        params = [torch.stack(t) for t in zip(
            *[self.small_sample_ppca(x[component_samples[i]], n_factors=l) for i in range(K)])]

        self.mu.data = params[0]
        self.W.data = params[1]
        self.log_Psi.data = params[2]


    def fit(self, x, max_iterations=20, feature_sampling=False):
        """
        Estimate maximum-likelihood MPPCA paramters for the complete dataset 
        using the Expectation Maximization algorithm from Tipping and Bishop 
        (1999) [2].

        Parameters:
        x (torch.Tensor): [n x d] tensor of input data
        max_iterations (int): the number of EM algorithm iterations
        feature_sampling (float or False): if float, denotes the fraction of
            total features to sample to speed up responsibility calculation

        Returns:
        lls (list): the per-iteration average log-likelihood values
        """
        assert not feature_sampling or type(feature_sampling) == float, 'set to desired sampling ratio'
        K, d, l = self.W.shape
        N = x.shape[0]

        x = x.to(self.mu.device)
        print('Initializing parameter values...')
        init_samples_per_component = (l+1)*2 if self.init_method == 'rnd_samples' else (l+1)*10
        self.init_from_data(x, samples_per_component=init_samples_per_component,
                             feature_sampling=feature_sampling)
        print('Init log-likelihood ={:.4f}'.format(torch.mean(self.per_sample_log_likelihood(x)).item()))

        # All equations and appendices reference Tipping and Bishop (1999) [2]
        def per_component_m_step(i):
            # (C.8)
            mui_new = torch.sum(r[:, [i]] * x, dim=0) / r_sum[i]
            sigma2_I = torch.exp(self.log_Psi[i, 0]) * torch.eye(l, device=x.device)
            inv_Mi = torch.inverse(self.W[i].T @ self.W[i] + sigma2_I)
            x_c = x - mui_new.reshape(1, d)
            # efficiently calculate (Si)(Wi) as discussed in Appendix C
            SiWi = (1.0/r_sum[i]) * (r[:, [i]]*x_c).T @ (x_c @ self.W[i])
            # (C.14)
            Wi_new = SiWi @ torch.inverse(sigma2_I + inv_Mi @ self.W[i].T @ SiWi)
            # (C.15)
            t1 = torch.trace(Wi_new.T @ (SiWi @ inv_Mi))
            trace_Si = torch.sum(N/r_sum[i] * torch.mean(r[:, [i]]*x_c*x_c, dim=0))
            sigma_2_new = (trace_Si - t1)/d

            return mui_new, Wi_new, torch.log(sigma_2_new) * torch.ones_like(self.log_Psi[i])

        lls = []
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
            ll = torch.mean(self.per_sample_log_likelihood(x)).item()
            print('Iteration {}/{}, train log-likelihood = {:.4f}, took {:.4f} sec'.format(
                it, max_iterations, ll, time.time()-t))
            lls.append(ll)
        return lls


    def batch_fit(self, train_dataset, test_dataset=None, batch_size=1000, test_size=1000, 
                  max_iterations=20, feature_sampling=False):
        """
        Estimate maximum-likelihood MPPCA paramters for the complete dataset 
        using the Expectation Maximization algorithm from Tipping and Bishop 
        (1999) [2]. This is a memory-efficient batched implementation for large 
        datasets that do not fit in memory:
        1) E step:
            For all mini-batches:
            - Calculate and store responsibilities
            - Accumulate sufficient statistics
        2) M step: 
            Re-calculate all parameters

        Parameters:
        train_dataset (torch Dataset): Dataset object containing the training data
        test_dataset (torch Dataset): Dataset object containing the test data 
            (if not provided, the training dataset will be used to compute the test loss)
        batch_size (int): the batch size
        test_size (int): the number of samples to use when reporting the likelihood
        max_iterations (int): the number of EM algorithm iterations (analogous 
            to epochs in the batch case)
        feature_sampling (float or False): if float, denotes the fraction of
            total features to sample to speed up responsibility calculation

        Returns:
        lls (list): the per-iteration average log-likelihood values
       """
        assert not feature_sampling or type(feature_sampling) == float, 'set to desired sampling ratio'
        K, d, l = self.W.shape

        init_samples_per_component = (l+1)*2 if self.init_method == 'rnd_samples' else (l+1)*10
        print('Random initialization using {}...'.format(self.init_method))
        init_keys = [key for i, key in enumerate(RandomSampler(train_dataset)) if i < init_samples_per_component*K]
        init_samples, _ = zip(*[train_dataset[key] for key in init_keys])
        self.init_from_data(torch.stack(init_samples).to(self.mu.device),
                             samples_per_component=init_samples_per_component,
                             feature_sampling=feature_sampling)

        # Read some test samples for test likelihood calculation
        test_dataset = test_dataset or train_dataset
        test_samples, _ = zip(*[test_dataset[key] for key in RandomSampler(test_dataset, num_samples=test_size, replacement=False)])
        test_samples = torch.stack(test_samples).to(self.mu.device)

        lls = []
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        for it in range(max_iterations):
            t = time.time()

            # Sufficient statistics
            sum_r = torch.zeros(size=[K], dtype=torch.float64, device=self.mu.device)
            sum_r_x = torch.zeros(size=[K, d], dtype=torch.float64, device=self.mu.device)
            sum_r_x_x_W = torch.zeros(size=[K, d, l], dtype=torch.float64, device=self.mu.device)
            sum_r_norm_x = torch.zeros(K, dtype=torch.float64, device=self.mu.device)

            lls.append(torch.mean(self.per_sample_log_likelihood(test_samples)).item())
            print('Iteration {}/{}, log-likelihood={:.4f}:'.format(it, max_iterations, lls[-1]))

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
                    sum_r_x_x_W[i] += (batch_r_x.T @ (batch_x @ self.W[i])).double()

            print(' / M...', end='', flush=True)
            self.pi_logits.data = torch.log(sum_r / torch.sum(sum_r)).float()
            self.mu.data = (sum_r_x / sum_r.reshape(-1, 1)).float()
            SW = sum_r_x_x_W / sum_r.reshape(-1, 1, 1) - \
                 (self.mu.reshape(K, d, 1) @ (self.mu.reshape(K, 1, d) @ self.W)).double()
            s2_I = torch.exp(self.log_Psi[:, 0]).reshape(K, 1, 1) * torch.eye(l, device=self.mu.device).reshape(1, l, l)
            M = (self.W.transpose(1, 2) @ self.W + s2_I).double()
            inv_M = torch.stack([torch.inverse(M[i]) for i in range(K)])   # (K, l, l)
            invM_AT_S_A = inv_M @ self.W.double().transpose(1, 2) @ SW   # (K, l, l)
            self.W.data = torch.stack([(SW[i] @ torch.inverse(s2_I[i].double() + invM_AT_S_A[i])).float()
                                       for i in range(K)])
            t1 = torch.stack([torch.trace(self.W[i].double().T @ (SW[i] @ inv_M[i])) for i in range(K)])
            t_s = sum_r_norm_x / sum_r - torch.sum(torch.pow(self.mu, 2.0), dim=1).double()
            self.log_Psi.data = torch.log((t_s - t1)/d).float().reshape(-1, 1) * torch.ones_like(self.log_Psi)

            print(' ({:.4f} sec)'.format(time.time()-t))

        lls.append(torch.mean(self.per_sample_log_likelihood(test_samples)).item())
        print('\nFinal train log-likelihood={:.4f}:'.format(lls[-1]))

        return lls


    def sgd_mfa_train(self, train_dataset, test_dataset=None, batch_size=256, test_size=1000, max_epochs=10,
                      learning_rate=0.0001, feature_sampling=False):
        """
        Stochastic gradient descent training of MFA after initialization using 
        MPPCA Expectation Maximization.

        Parameters:
        train_dataset (torch Dataset): Dataset object containing the training data
        test_dataset (torch Dataset): Dataset object containing the test data 
            (if not provided, the training dataset will be used to compute the test loss)
        batch_size (int): the batch size
        test_size (int): the number of samples to use when reporting the likelihood
        max_epochs (int): the number of training epochs
        learning_rate (float): the optimizer learning rate
        feature_sampling (float or False): if float, denotes the fraction of
            total features to sample to speed up responsibility calculation

        Returns:
        lls (list): the per-iteration average log-likelihood values
        """
        assert not torch.all(self.W == 0.), 'SGD MFA training requires initialization.'
        assert not feature_sampling or type(feature_sampling) == float, 'set to desired sampling ratio'

        self.mu.requires_grad = self.W.requires_grad = self.log_Psi.requires_grad = True
        K, d, l = self.W.shape

        # Read some test samples for test likelihood calculation
        test_dataset = test_dataset or train_dataset
        test_samples, _ = zip(*[test_dataset[key] for key in RandomSampler(test_dataset, num_samples=test_size, replacement=False)])
        test_samples = torch.stack(test_samples).to(self.mu.device)

        # main training loop
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        lls = []
        self.train()
        for epoch in range(max_epochs):
            t = time.time()
            for idx, (batch_x, _) in enumerate(loader):
                if idx > 0 and idx%100 == 0:
                    print("Iteration {}".format(idx))
                    print("Loss {:.8f}".format(torch.mean(self.per_sample_log_likelihood(test_samples)).item()))
                sampled_features = np.random.choice(d, int(d*feature_sampling)) if feature_sampling else None
                batch_x = batch_x.to(self.mu.device)
                optimizer.zero_grad()
                loss = -torch.mean(self.per_sample_log_likelihood(batch_x, sampled_features=sampled_features))
                loss.backward()
                optimizer.step()
            lls.append(torch.mean(self.per_sample_log_likelihood(test_samples)).item())
            print('\nEpoch {}: Test ll = {:.4f} ({:.4f} sec)'.format(epoch, lls[-1], time.time()-t))
        self.pi_logits.requires_grad = self.mu.requires_grad = self.W.requires_grad = self.log_Psi.requires_grad = False

        return lls