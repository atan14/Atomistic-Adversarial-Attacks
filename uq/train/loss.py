import torch
import numpy as np


class EvidentialLoss:
    def __init__(
        self,
        lamb=0.2,
        epsilon=0,
        output="energy",
        reduce_mean=False,
        clamp_min=1e-4,
        **kwargs
    ):
        self.lamb = lamb
        self.epsilon = epsilon
        self.output = output
        self.reduce_mean = reduce_mean
        self.clamp_min = clamp_min

    def __call__(self, targets, predicted):
        error = self.get_error(targets, predicted)
        v, alpha, beta = self.parse_uncertainty(predicted)

        nll_loss = self.nll_loss(error, v, alpha, beta)
        reg_loss = self.reg_loss(error, v, alpha)

        loss = nll_loss + self.lamb * (reg_loss - self.epsilon)
        return loss.mean()

    def get_error(self, targets, predicted):
        pred = predicted[self.output]
        targ = targets[self.output]
        if len(pred.shape) > len(targ.shape):
            targ = targ.unsqueeze(-1).expand_as(pred)

        error = targ - pred
        if self.output == "energy_grad":
            error = torch.norm(error, dim=1)

        return error.squeeze()

    def parse_uncertainty(self, predicted):
        v = predicted['v'].squeeze()
        alpha = predicted['alpha'].squeeze()
        beta = predicted['beta'].squeeze()

        if v.shape != alpha.shape:
            num_atoms = torch.tensor([alpha.shape[0] // v.shape[0]] * v.shape[0])
            v = torch.repeat_interleave(v, num_atoms)

        v, alpha, beta = self.clamp(v), self.clamp(alpha), self.clamp(beta)
        return v, alpha, beta

    def clamp(self, x):
        return x + self.clamp_min

    def get_evidential_uncertainty(self, predicted):
        v, alpha, beta = self.parse_uncertainty(predicted)
        return {
            "aleatoric": beta / (alpha - 1),
            "epistemic": beta / (v * (alpha - 1)),
        }

    def nll_loss(self, error, v, alpha, beta):
        twoBlambda = 2 * beta * (1 + v)

        nll = (
            0.5 * torch.log(np.pi / v)
            - alpha * torch.log(twoBlambda)
            + (alpha + 0.5) * torch.log(v * error**2 + twoBlambda)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        L_NLL = torch.mean(nll, dim=-1) if self.reduce_mean else nll
        return L_NLL

    def reg_loss(self, error, v, alpha):
        reg = error * (2 * v + alpha)
        L_REG = torch.mean(reg, dim=-1) if self.reduce_mean else reg
        return L_REG


class MaeLoss:
    def __init__(self, output="energy", **kwargs):
        self.output = output

    def __call__(self, targets, predicted):
        pred = predicted[self.output]
        targ = targets[self.output]
        if len(pred.shape) > len(targ.shape):
            targ = targ.unsqueeze(-1).expand_as(pred)
        return self.loss_fn(pred, targ)

    def loss_fn(self, pred, targ):
        loss = (targ - pred).abs().mean()
        return loss


class MseLoss:
    def __init__(self, output="energy", **kwargs):
        self.output = output

    def __call__(self, targets, predicted):
        pred = predicted[self.output]
        targ = targets[self.output]
        if len(pred.shape) > len(targ.shape):
            targ = targ.unsqueeze(-1).expand_as(pred)
        return self.loss_fn(pred, targ)

    def loss_fn(self, pred, targ):
        loss = ((targ - pred) ** 2).sum() ** 0.5
        return loss


class NllLoss:
    def __init__(self, output="energy", **kwargs):
        self.output = output

    def __call__(self, targets, predicted):
        pred = predicted[self.output]
        targ = targets[self.output]
        if len(pred.shape) > len(targ.shape):
            targ = targ.unsqueeze(-1).expand_as(pred)
        pred_var = predicted['var']

        loss = self.loss_fn(pred, targ, pred_var)

        return loss

    def loss_fn(self, pred, targ, pred_var):
        clamped_var = torch.clamp(pred_var, min=1e-8)
        error = targ - pred
        error = torch.norm(error, dim=1) if self.output == "energy_grad" else error
        return (
            (torch.log(2 * np.pi * clamped_var) * (1 / 2))
            + (error**2) / (2 * clamped_var)
        ).mean()


class GmmNllLoss:
    def __init__(self, n_clusters, *args, **kwargs):
        """
        Args:
            gm_model (sklearn.mixture.GaussianMixture): Gaussian Mixture model to predict NLL
        """
        super().__init__(*args, **kwargs)
        self.n_clusters = n_clusters

    def fit_gaussian_mixture(self, embedding):
        from sklearn.mixture import GaussianMixture
        self.gm_model = GaussianMixture(n_components=self.n_clusters)
        self.gm_model.fit(embedding)

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

    def __call__(self, train_predicted, test_predicted):
        train_embedding = train_predicted['embedding']
        test_embedding = test_predicted['embedding']
        device = test_embedding.device

        self.fit_gaussian_mixture(train_embedding.squeeze().detach().cpu())
        means = self.gm_model.means_
        precisions_cholesky = self.gm_model.precisions_cholesky_
        weights = self.gm_model.weights_

        means, precisions_cholesky, weights = self.check_inputs(means, precisions_cholesky, weights)

        return self.nll(test_embedding, means, precisions_cholesky, weights, device=device)


class CombinedLoss:
    def __init__(self, energy_loss, forces_loss, energy_coef=0.1, forces_coef=1):
        self.e_loss = energy_loss
        self.f_loss = forces_loss
        self.e_coef = energy_coef
        self.f_coef = forces_coef

    def __call__(self, batch, pred):
        return (self.e_coef * self.e_loss(batch, pred)) + (
            self.f_coef * self.f_loss(batch, pred)
        )
