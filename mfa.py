import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time
import warnings


class MFA(torch.nn.Module):
    """
    A class representing a Mixture of Factor Analyzers [1] / Mixture of Probabilistic PCA [2] in pytorch.
    MFA/MPPCA are Gaussian Mixture Models with low-rank-plus-diagonal covariance, enabling efficient modeling
    of high-dimensional domains in which the data resides near lower-dimensional subspaces.
    The implementation is based on [3] (please quote this if you are using this package in your research).

    Attributes (model parameters):
    ------------------------------
    MU: Tensor shaped [n_components, n_features]
        The component means.
    A: Tensor shaped [n_components, n_features, n_factors]
        The component subspace directions / factor loadings. These should be orthogonal (but not orthonormal)
    lod_D: Tensor shaped [n_components, n_features]
        Log of the component diagonal variance values. Note that in MPPCA, all values along the diagonal are the same.
    PI_logits: Tensor shaped [n_components]
        Log of the component mixing-coefficients (probabilities). Apply softmax to get the actual PI values.

    Main Methods:
    -------------
    fit:
        Fit the MPPCA model parameters to pre-loaded training data using EM

    batch_fit:
        Fit the MPPCA model parameters to a (possibly large) pytorch dataset using EM in batches

    sample:
        Generate new samples from the trained model

    per_component_log_likelihood, log_prob, log_likelihood:
        Probability query methods

    responsibilities, log_responsibilities, map_component:
        Responsibility (which component the sample comes from) query methods

    reconstruct, conditional_reconstruct:
        Reconstruction and in-painting

    [1] Tipping, Michael E., and Christopher M. Bishop. "Mixtures of probabilistic principal component analyzers."
        Neural computation 11.2 (1999): 443-482.
    [2] Ghahramani, Zoubin, and Geoffrey E. Hinton. "The EM algorithm for mixtures of factor analyzers."
        Vol. 60. Technical Report CRG-TR-96-1, University of Toronto, 1996.
    [3] Richardson, Eitan, and Yair Weiss. "On gans and gmms."
        Advances in Neural Information Processing Systems. 2018.

    """
    def __init__(self, n_components, n_features, n_factors, isotropic_noise=True, init_method='rnd_samples'):
        super(MFA, self).__init__()
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
        """
        Generate random samples from the trained MFA / MPPCA
        :param n: How many samples
        :param with_noise: Add the isotropic / diagonal noise to the generated samples
        :return: samples [n, n_features], c_nums - originating component numbers
        """
        if torch.all(self.W == 0.):
            warnings.warn('SGD MFA training requires initialization. Please call batch_fit() first.')

        K, d, l = self.W.shape
        c_nums = np.random.choice(K, n, p=torch.softmax(self.pi_logits, dim=0).detach().cpu().numpy())
        z_l = torch.randn(n, l, device=self.W.device)
        z_d = torch.randn(n, d, device=self.W.device) if with_noise else torch.zeros(n, d, device=self.W.device)
        samples = torch.stack([self.W[c_nums[i]] @ z_l[i] + self.mu[c_nums[i]] + z_d[i] * torch.exp(0.5*self.log_Psi[c_nums[i]])
                               for i in range(n)])
        return samples, c_nums

    @staticmethod
    def _component_log_likelihood(x, PI, MU, A, log_D):
        K, d, l = A.shape
        AT = A.transpose(1, 2)
        iD = torch.exp(-log_D).view(K, d, 1)
        # L is IPiDP; Woodbury matrix lemm, det(A + UCV) = det(C^-1 + VA^-1U) det(C) det(A) where C and A are I
        L = torch.eye(l, device=A.device).reshape(1, l, l) + AT @ (iD*A)
        # L = M / sigma^2? yeah, 1/posterior cov matrix
        iL = torch.inverse(L)
        # this is posterior cov matrix

        def per_component_md(i):
            x_c = (x - MU[i].reshape(1, d)).T  # shape = (d, n)
            # last paragraph, plus 3.5 vs 3.7. NO! MFA paper, pg. 3. Between eq. 3 and 4.
            m_d_1 = (iD[i] * x_c) - ((iD[i] * A[i]) @ iL[i]) @ (AT[i] @ (iD[i] * x_c))
            return torch.sum(x_c * m_d_1, dim=0)

        m_d = torch.stack([per_component_md(i) for i in range(K)])
        det_L = torch.logdet(L)
        log_det_Sigma = det_L - torch.sum(torch.log(iD.reshape(K, d)), axis=1)
        log_prob_data_given_components = -0.5 * ((d*np.log(2.0*torch.pi) + log_det_Sigma).reshape(K, 1) + m_d)
        return PI.reshape(1, K) + log_prob_data_given_components.T  # (eq) C.1


    def per_component_log_likelihood(self, x, sampled_features=None):
        """
        Calculate per-sample and per-component log-likelihood values
        :param x: samples [n, n_features]
        :param sampled_features: list of feature coordinates to use
        :return: log-probability values [n, n_components]
        """
        if sampled_features is not None:
            return MFA._component_log_likelihood(x[:, sampled_features], torch.softmax(self.pi_logits, dim=0),
                                                 self.mu[:, sampled_features],
                                                 self.W[:, sampled_features],
                                                 self.log_Psi[:, sampled_features])
        return MFA._component_log_likelihood(x, torch.softmax(self.pi_logits, dim=0), self.mu, self.W, self.log_Psi)

    def log_prob(self, x, sampled_features=None):
        """
        Calculate per-sample log-probability values
        :param x: samples [n, n_features]
        :param sampled_features: list of feature coordinates to use
        :return: log-probability values [n]
        """
        return torch.logsumexp(self.per_component_log_likelihood(x, sampled_features), dim=1)

    def log_likelihood(self, x, sampled_features=None):
        """
        Calculate the log-likelihood of the given data
        :param x: samples [n, n_features]
        :param sampled_features: list of feature coordinates to use
        :return: (total) log-likelihood value
        """
        return torch.sum(self.log_prob(x, sampled_features))

    def log_responsibilities(self, x, sampled_features=None):
        """
        Calculate the log-responsibilities (log of the responsibility values - probability of each sample to originate
        from each of the component.
        :param x: samples [n, n_features]
        :param sampled_features: list of feature coordinates to use
        :return: log-responsibilities values [n, n_components]
        """
        comp_LLs = self.per_component_log_likelihood(x, sampled_features)
        return comp_LLs - torch.logsumexp(comp_LLs, dim=1).reshape(-1, 1)

    def responsibilities(self, x, sampled_features=None):
        """
        Calculate the responsibilities - probability of each sample to originate from each of the component.
        :param x: samples [n, n_features]
        :param sampled_features: list of feature coordinates to use
        :return: responsibility values [n, n_components]
        """
        return torch.exp(self.log_responsibilities(x, sampled_features))

    def map_component(self, x, sampled_features=None):
        """
        Get the Maximum a Posteriori component numbers
        :param x: samples [n, n_features]
        :param sampled_features: list of feature coordinates to use
        :return: component numbers [n]
        """
        return torch.argmax(self.log_responsibilities(x, sampled_features), dim=1)

    def conditional_reconstruct(self, full_x, observed_features):
        """
        Calculates the mean of the conditional probability P(x_h | x_o)
        References:
        https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf#subsubsection.8.1.3
        https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        Note: This is equivalent to calling reconstruct with sampled_features

        :param full_x: the full vectors (including the hidden coordinates, which can contain any values)
        :param observed_features: tensor containing a list of the observed coordinates of x
        :return: A cloned version of full_x with the hidden features reconstructed
        """
        assert observed_features is not None
        K, d, l = self.W.shape
        c_i = self.map_component(full_x, observed_features)

        mask = torch.zeros(d, dtype=bool)
        mask[observed_features] = True

        A_a = self.W[c_i][:, ~mask, :]
        A_b = self.W[c_i][:, mask, :]
        MU_a = self.mu[c_i][:, ~mask]
        MU_b = self.mu[c_i][:, mask]
        iD_b = torch.exp(-self.log_Psi[c_i][:, mask]).unsqueeze(2)

        iL_b = torch.inverse(torch.eye(l, device=MU_b.device).reshape(1, l, l) + A_b.transpose(1, 2) @ (iD_b*A_b))
        x_b_l = ((A_b * iD_b).transpose(1,2) @ (full_x[:, mask] - MU_b).unsqueeze(2))
        x_hat = full_x.clone()
        x_hat[:, ~mask] =  (MU_a.unsqueeze(2) + A_a @ x_b_l - A_a @ (A_b.transpose(1, 2) @
                                                                     (iD_b * (A_b @ iL_b @ x_b_l)))).squeeze(dim=2)
        return x_hat

    def reconstruct(self, full_x, observed_features=None):
        """
        Reconstruct samples from the model - find the MAP component and latent z for each sample and regenerate

        :param full_x: the full vectors (including the hidden coordinates, which can contain any values)
        :param observed_features: tensor containing a list of the observed coordinates of x
        :return: Reconstruction of full_x based on the trained model and observed features
        """
        K, d, l = self.W.shape
        c_i = self.map_component(full_x, observed_features)

        used_features = observed_features if observed_features is not None else torch.arange(0, d)
        x = full_x[:, used_features]
        MU = self.mu[:, used_features]
        A = self.W[:, used_features]
        AT = A.transpose(1, 2)
        iD = torch.exp(-self.log_Psi[:, used_features]).unsqueeze(2)
        L = torch.eye(l, device=MU.device).reshape(1, l, l) + AT @ (iD*A)
        iL = torch.inverse(L)

        # per eq. 2 in Ghahramani and Hinton 1996 + the matrix inversion lemma (also described there).
        x_c = (x - MU[c_i]).unsqueeze(2)
        iD_c = iD[c_i]
        m_d_1 = (iD_c * x_c) - ((iD_c * A[c_i]) @ iL[c_i]) @ (AT[c_i] @ (iD_c * x_c))
        mu_z = AT[c_i] @ m_d_1
        return (self.W[c_i] @ mu_z).reshape(-1, d) + self.mu[c_i]

    @staticmethod
    def _small_sample_ppca(x, n_factors):
        # See https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
        mu = torch.mean(x, dim=0)
        # U, S, V = torch.svd(x - mu.reshape(1, -1))    # torch svd is less memory-efficient
        U, S, V = np.linalg.svd((x - mu.reshape(1, -1)).cpu().numpy(), full_matrices=False)
        V = torch.from_numpy(V.T).to(x.device)
        S = torch.from_numpy(S).to(x.device)
        # (eq) 3.13
        sigma_squared = torch.sum(torch.pow(S[n_factors:], 2.0))/((x.shape[0]-1) * (x.shape[1]-n_factors))
        # (eq) 3.12, S in (eq) 3.9
        A = V[:, :n_factors] * torch.sqrt((torch.pow(S[:n_factors], 2.0).reshape(1, n_factors)/(x.shape[0]-1) - sigma_squared))
        return mu, A, torch.log(sigma_squared) * torch.ones(x.shape[1], device=x.device)

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
            *[MFA._small_sample_ppca(x[component_samples[i]], n_factors=l) for i in range(K)])]

        self.mu.data = params[0]
        self.W.data = params[1]
        self.log_Psi.data = params[2]

    def _parameters_sanity_check(self):
        K, d, l = self.W.shape
        assert torch.all(torch.softmax(self.pi_logits, dim=0) > 0.01/K), self.pi_logits
        assert torch.all(torch.exp(self.log_Psi) > 1e-5) and torch.all(torch.exp(self.log_Psi) < 1.0), \
            '{} - {}'.format(torch.min(self.log_Psi).item(), torch.max(self.log_Psi).item())
        #assert torch.all(torch.abs(self.W) < 10.0), torch.max(torch.abs(self.W))
        #assert torch.all(torch.abs(self.mu) < 1.0), torch.max(torch.abs(self.mu))

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

        x = x.to(self.mu.device)
        
        print('Random init...')
        init_samples_per_component = (l+1)*2 if self.init_method == 'rnd_samples' else (l+1)*10
        self._init_from_data(x, samples_per_component=init_samples_per_component,
                             feature_sampling=feature_sampling/2 if feature_sampling else False)
        print('Init log-likelihood =', round(torch.mean(self.log_prob(x)).item(), 1))

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
            ll = round(torch.mean(self.log_prob(x)).item(), 1) #if it % 5 == 0 else '.....'
            print('Iteration {}/{}, train log-likelihood = {}, took {:.1f} sec'.format(it, max_iterations, ll,
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
        test_samples, _ = zip(*[test_dataset[key] for key in RandomSampler(test_dataset, num_samples=test_size, replacement=True)])
        #test_dataset = test_dataset or train_dataset
        #all_test_keys = [key for key in SequentialSampler(test_dataset)]
        #test_samples, _ = zip(*[test_dataset[key] for key in all_test_keys[:test_size]])
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

            self._parameters_sanity_check()
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
                loss = -self.log_likelihood(batch_x, sampled_features=sampled_features) / batch_size
                #loss = -torch.mean(self.log_prob(test_samples, sampled_features=sampled_features))
                loss.backward()
                optimizer.step()
            ll_log.append(torch.mean(self.log_prob(test_samples)).item())
            print('\nEpoch {}: Test ll = {} ({} sec)'.format(epoch, ll_log[-1], time.time()-t))
            self._parameters_sanity_check()
        self.pi_logits.requires_grad = self.mu.requires_grad = self.W.requires_grad = self.log_Psi.requires_grad = False
        return ll_log