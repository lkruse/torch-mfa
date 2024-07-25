import torch
import time
from sklearn.cluster import KMeans

class MPPCA(torch.nn.Module):

    def __init__(self, n_components, n_features, n_factors):
        """
        Initialize the Mixture of Probabilistic Principal Component Analyzers.

        Original publication:
        [1] Tipping, M. E., & Bishop, C. M. (1999). Mixtures of probabilistic 
            principal component analyzers. Neural Computation, 11(2), 443-482.

        The implementation is based on the open-source code from
        [2] Richardson, E., & Weiss, Y. (2018). On GANs and GMMs. Advances in 
            Neural Information Processing Systems, 31.

        Parameters:
            waypoints (list of np.array): list of waypoints as numpy arrays
            stop_duration (float): duration of pallet pause at each waypoint (seconds)
            max_speed (float): maximum speed of the pallet (meters per second)
            acc (float): acceleration of the pallet (meters per second^2)
            dt (float): simulation timestep (seconds)
        """
        super(MPPCA, self).__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.n_factors = n_factors

        self.mu = torch.zeros(n_components, n_features)
        self.W = torch.zeros(n_components, n_features, n_factors)
        self.sigma2 = torch.ones(n_components)
        self.pi = torch.ones(n_components)/float(n_components)


    def sample(self, n, with_noise=True):
        K, d, l = self.W.shape
        sampled_components = torch.multinomial(self.pi, n, replacement=True)
        z_l = torch.randn(n, l, device=self.W.device)

        if with_noise:
            z_d = torch.randn(n, d, device=self.W.device)  
        else:
            z_d = torch.zeros(n, d, device=self.W.device)
        
        Wz = self.W[sampled_components] @ z_l[..., None]
        mu = self.mu[sampled_components][..., None]
        epsilon = (z_d * self.sigma2[sampled_components][..., None]**0.5)[..., None]

        samples = Wz + mu + epsilon

        return samples.squeeze(), sampled_components


    def _component_log_likelihood(self, x, W, mu, pi, sigma2):
        K, d, l = W.shape
        WT = W.transpose(1,2)
        inv_sigma2 = (1.0/sigma2 * torch.ones(d,1)).view(d,K,1).transpose(0,1)
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


    def log_prob(self, x):
        """
        Calculate per-sample log-probability values
        :param x: samples [n, n_features]
        :return: log-probability values [n]
        """
        component_lls = self._component_log_likelihood(x, self.W, self.mu, self.pi, self.sigma2)

        return torch.logsumexp(component_lls, dim=1)
    

    def responsibilities(self, x):
        """
        Calculate the responsibilities - probability of each sample to originate from each of the component.
        :param x: samples [n, n_features]
        :return: responsibility values [n, n_components]
        """
        component_lls = self._component_log_likelihood(x, self.W, self.mu, self.pi, self.sigma2)
        log_responsibilities = component_lls - self.log_prob(x).reshape(-1, 1)

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

        return W, mu, sigma2


    def _init_from_data(self, x):
        K, d, l = self.W.shape
        t = time.time()
        print('Performing K-means clustering to {} clusters...'.format(K))
        clusters = KMeans(n_clusters=K, max_iter=300).fit(x)
        print('... took {:.4f} sec'.format(time.time() - t))
        labels = [clusters.labels_ == i for i in range(K)]

        params = [torch.stack(t) for t in zip(
            *[self._small_sample_ppca(x[labels[i]], n_factors=l) for i in range(K)])]

        self.W = params[0]
        self.mu = params[1]
        self.sigma2 = params[2]


    def fit(self, x, max_iterations=50):
        K, d, l = self.W.shape
        N = x.shape[0]

        self._init_from_data(x)
        print('Initial log-likelihood = {:.4f}'.format(torch.mean(self.log_prob(x)).item()))

        # All equations and appendices reference Tipping and Bishop (1999) [1]
        def per_component_m_step(i):
            # (C.8)
            mui_new = torch.sum(r[:, [i]] * x, dim=0) / r_sum[i]
            sigma2_I = self.sigma2[i] * torch.eye(l, device=x.device)
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

            return Wi_new, mui_new, sigma_2_new

        log_likelihoods = []
        for it in range(max_iterations):
            t = time.time()
            r = self.responsibilities(x)
            r_sum = torch.sum(r, dim=0)
            new_params = [torch.stack(t) for t in zip(*[per_component_m_step(i) for i in range(K)])]
            self.W = new_params[0]
            self.mu = new_params[1]
            self.sigma2 = new_params[2]
            self.pi = r_sum / torch.sum(r_sum)
            ll = torch.mean(self.log_prob(x)).item()
            print('Iteration {}/{}, train log-likelihood = {:.4f}, took {:.4f} sec'.format(it, max_iterations, ll,
                                                                                   time.time()-t))
            log_likelihoods.append(ll)

        return log_likelihoods