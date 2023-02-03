import torch
from evi.data import Dataset


class AdvLoss:
    def __init__(self, train: Dataset, temperature: float = 1, **kwargs):
        self.e = train.props["energy"] / train.props['num_atoms']
        self.temperature = temperature

    def boltzmann_probability(self, e):
        return torch.exp(-e / self.temperature)

    @property
    def partition_fn(self):
        return self.boltzmann_probability(self.e).mean()

    def probability_fn(self, yp):
        return self.boltzmann_probability(yp) / self.partition_fn

    def split(self, y, num_atoms):
        split_y = torch.split(y, list(num_atoms))
        sum_split = torch.stack(split_y, dim=0)
        return sum_split

    def loss_fn(self, results, **kwargs):
        return NotImplementedError

    def __call__(self, results, **kwargs):
        return self.loss_fn(results, **kwargs).sum()


class AdvEvidential(AdvLoss):
    def __init__(
        self, uncertainty_source="epistemic", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.uncertainty_source = uncertainty_source

    def uncertainty_fn(self, v, alpha, beta):
        if self.uncertainty_source == "aleatoric":
            return beta / (alpha - 1)
        if self.uncertainty_source == "epistemic":
            return beta / (v * (alpha - 1))

    def reshape_parameters(self, results, num_atoms=None):
        v = results['v'].squeeze()
        alpha = results['alpha'].squeeze()
        beta = results['beta'].squeeze()
        if v.shape[0] / results['energy'].shape[0] != 1.:
            v = self.split(v, list(num_atoms))
            v = v.mean(-1, keepdims=True)
        if alpha.shape[0] / results['energy'].shape[0] != 1.:
            alpha = self.split(alpha, list(num_atoms))
            alpha = alpha.mean(-1, keepdims=True)
        if beta.shape[0] / results['energy'].shape[0] != 1.:
            beta = self.split(beta, list(num_atoms))
            beta = beta.mean(-1, keepdims=True)
        return v, alpha, beta

    def clamp(self, v, alpha, beta, min_value=None):
        # add numerical stability to denominator values if needed
        if min_value:
            v = v + min_value
            alpha = alpha + min_value
        return v, alpha, beta

    def loss_fn(self, results, num_atoms=None, min_value=None, **kwargs):
        v, alpha, beta = self.reshape_parameters(results, num_atoms)
        v, alpha, beta = self.clamp(v, alpha, beta, min_value)
        uncertainty = self.uncertainty_fn(v, alpha, beta)
        energy_per_atom = (results['energy'].mean(-1) / num_atoms).reshape(-1, 1)
        probability_fn = self.probability_fn(energy_per_atom)
        return -uncertainty * probability_fn


class AdvMVE(AdvLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss_fn(self, results, num_atoms=None, **kwargs):
        var = results['var']
        if var.shape[0] / results['energy'].shape[0] != 1.:
            var = self.split(var, num_atoms)
            var = var.squeeze()
        uncertainty = var.mean(-1, keepdims=True)
        energy_per_atom = (results['energy'].mean(-1) / num_atoms).reshape(-1, 1)
        probability_fn = self.probability_fn(energy_per_atom)
        return -uncertainty * probability_fn


class AdvEnsemble(AdvLoss):
    def __init__(self, q="energy_grad", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q = q

    def loss_fn(self, results, num_atoms=None, **kwargs):
        q = results[self.q]
        if self.q == "energy_grad":
            q = self.split(q, num_atoms)
            q = q.var(-1)
            q = torch.norm(q, dim=-1)
        if self.q == 'energy':
            q = q.var(-1)
        uncertainty = q.mean(-1, keepdims=True)
        energy_per_atom = (results['energy'].mean(-1) / num_atoms).reshape(-1, 1)
        probability_fn = self.probability_fn(energy_per_atom)
        return -uncertainty * probability_fn


class AdvGMM(AdvLoss):
    def __init__(self, gm_model=None, *args, **kwargs):
        """
        Args:
            gm_model (sklearn.mixture.GaussianMixture): Gaussian Mixture model to predict NLL
        """
        super().__init__(*args, **kwargs)
        self.gm_model = gm_model

    def check_tensors(self, embedding, means, precisions_cholesky, weights, device):
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)
        if not isinstance(means, torch.Tensor):
            means = torch.tensor(means)
        if not isinstance(precisions_cholesky, torch.Tensor):
            precisions_cholesky = torch.tensor(precisions_cholesky)
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)

        embedding = embedding.squeeze().double().to(device)
        means = means.double().to(device)
        precisions_cholesky = precisions_cholesky.double().to(device)
        weights = weights.double().to(device)

        return embedding, means, precisions_cholesky, weights

    def check_inputs(self, means, precisions_cholesky, weights):
        if (self.gm_model is None) and (means is None or precisions_cholesky is None or weights is None):
            raise Exception("AdvGM: Input not complete for prediction")
        if self.gm_model is not None:
            if means is None:
                means = self.gm_model.means_
            if precisions_cholesky is None:
                precisions_cholesky = self.gm_model.precisions_cholesky_
            if weights is None:
                weights = self.gm_model.weights_
        return means, precisions_cholesky, weights

    def estimate_log_prob(self, embedding, means, precisions_cholesky):
        n_samples, n_features = embedding.shape
        n_clusters, _ = means.shape

        log_det = torch.sum(
            torch.log(precisions_cholesky.reshape(n_clusters, -1)[:, ::n_features+1]), dim=1,
        )

        log_prob = torch.empty((n_samples, n_clusters)).to(embedding.device)
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_cholesky)):
            y = torch.matmul(embedding, prec_chol) - (mu.reshape(1, -1) @ prec_chol).squeeze()
            log_prob[:, k] = torch.sum(torch.square(y), dim=1)
        log2pi = torch.log(torch.tensor([2 * torch.pi])).to(embedding.device)
        return -0.5 * (n_features * log2pi + log_prob) + log_det

    def nll(self, embedding, means, precisions_cholesky, weights, device):
        embedding, means, precisions_cholesky, weights = self.check_tensors(embedding, means, precisions_cholesky, weights, device)

        log_prob = self.estimate_log_prob(embedding, means, precisions_cholesky)
        log_weights = torch.log(weights)
        weighted_log_prob = log_prob + log_weights

        weighted_log_prob_max = weighted_log_prob.max(axis=1).values
        # logsumexp is numerically unstable for big arguments
        # below, the calculation below makes it stable
        # log(sum_i(a_i)) = log(exp(a_max) * sum_i(exp(a_i - a_max))) = a_max + log(sum_i(exp(a_i - a_max)))
        wlp_stable = weighted_log_prob - weighted_log_prob_max.reshape(-1, 1)
        logsumexp = weighted_log_prob_max + torch.log(torch.sum(torch.exp(wlp_stable), dim=1))
        return -logsumexp

    def loss_fn(self, results, num_atoms=None, **kwargs):
        embedding = results['embedding']
        energy = results['energy']
        means = results.get("means", None)
        precisions_cholesky = results.get("precisions_cholesky", None)
        weights = results.get("weights", None)

        means, precisions_cholesky, weights = self.check_inputs(means, precisions_cholesky, weights)

        uncertainty = self.nll(embedding, means, precisions_cholesky, weights, device=energy.device)
        energy_per_atom = (results['energy'].mean(-1) / num_atoms).reshape(-1, 1)
        probability_fn = self.probability_fn(energy_per_atom)
        return -uncertainty * probability_fn
