import torch
import time
from sklearn.cluster import KMeans

class MPPCA(torch.nn.Module):

    def __init__(self, n_components, n_features, n_factors):
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

        return samples, sampled_components


    def _component_log_likelihood(self, x, pi, mu, W, sigma2):
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
        return torch.logsumexp(self._component_log_likelihood(x, self.pi, self.mu, self.W, self.sigma2), dim=1)
    

    def responsibilities(self, x):
        """
        Calculate the responsibilities - probability of each sample to originate from each of the component.
        :param x: samples [n, n_features]
        :return: responsibility values [n, n_components]
        """
        comp_LLs = self._component_log_likelihood(x, self.pi, self.mu, self.W, self.sigma2)
        log_responsibilities = comp_LLs - self.log_prob(x).reshape(-1, 1)

        return torch.exp(log_responsibilities)


    def _small_sample_ppca(self, x, n_factors):
        # See https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
        mu = torch.mean(x, dim=0)
        U, S, V = torch.svd(x - mu.reshape(1, -1))

        V = V.T.to(x.device)
        S = S.to(x.device)
        # (eq) 3.13
        sigma_squared = torch.sum(torch.pow(S[n_factors:], 2.0))/((x.shape[0]-1) * (x.shape[1]-n_factors))
        # (eq) 3.12, S in (eq) 3.9
        A = V[:, :n_factors] * torch.sqrt((torch.pow(S[:n_factors], 2.0).reshape(1, n_factors)/(x.shape[0]-1) - sigma_squared))
        return mu, A, sigma_squared#torch.log(sigma_squared) * torch.ones(x.shape[1], device=x.device)


    def _init_from_data(self, x):
        n = x.shape[0]
        K, d, l = self.W.shape

        t = time.time()
        print('Performing K-means clustering of {} samples in dimension {} to {} clusters...'.format(
            x.shape[0], d, K))
        _x = x.cpu().numpy()
        clusters = KMeans(n_clusters=K, max_iter=300).fit(_x)
        print('... took {} sec'.format(time.time() - t))
        component_samples = [clusters.labels_ == i for i in range(K)]

        params = [torch.stack(t) for t in zip(
            *[self._small_sample_ppca(x[component_samples[i]], n_factors=l) for i in range(K)])]

        self.mu.data = params[0]
        self.W.data = params[1]
        #self.log_D.data = params[2]
        self.sigma2.data = params[2]


    def fit(self, x, max_iterations=20):
        K, d, l = self.W.shape
        N = x.shape[0]

        print('Random init...')
        self._init_from_data(x)
        print('Init log-likelihood =', round(torch.mean(self.log_prob(x)).item(), 1))

        def per_component_m_step(i):
            mu_i = torch.sum(r[:, [i]] * x, dim=0) / r_sum[i]
            s2_I = self.sigma2[i] * torch.eye(l, device=x.device)
            inv_M_i = torch.inverse(self.W[i].T @ self.W[i] + s2_I)
            x_c = x - mu_i.reshape(1, d)
            SiAi = (1.0/r_sum[i]) * (r[:, [i]]*x_c).T @ (x_c @ self.W[i])
            invM_AT_Si_Ai = inv_M_i @ self.W[i].T @ SiAi
            A_i_new = SiAi @ torch.inverse(s2_I + invM_AT_Si_Ai)
            t1 = torch.trace(A_i_new.T @ (SiAi @ inv_M_i))   # (eq) 6 in [2]
            trace_S_i = torch.sum(N/r_sum[i] * torch.mean(r[:, [i]]*x_c*x_c, dim=0)) # (eq) 6 in [2]
            sigma_2_new = (trace_S_i - t1)/d # (eq) 6 in [2]
            return mu_i, A_i_new, sigma_2_new#torch.log(sigma_2_new) * torch.ones_like(self.log_D[i])


        ll_log = []
        for it in range(max_iterations):
            t = time.time()
            r = self.responsibilities(x)
            r_sum = torch.sum(r, dim=0)
            new_params = [torch.stack(t) for t in zip(*[per_component_m_step(i) for i in range(K)])]
            self.mu.data = new_params[0]
            self.W.data = new_params[1]
            #self.log_D.data = new_params[2]
            self.sigma2.data = new_params[2]
            self.pi.data = r_sum / torch.sum(r_sum)
            ll = round(torch.mean(self.log_prob(x)).item(), 5)
            print('Iteration {}/{}, train log-likelihood = {}, took {:.1f} sec'.format(it, max_iterations, ll,
                                                                                   time.time()-t))
            ll_log.append(ll)
        return ll_log