import numpy as np
import torch


class Metric:
    r"""
    Base class for all metrics.

    Metrics measure the performance during the training and evaluation.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
            `MSE_[target]` will be used (Default: None)
    """

    def __init__(self, target, name=None):
        self.target = target
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

        self.loss = 0.0
        self.n_entries = 0.0

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        self.loss = 0.0
        self.n_entries = 0.0

    def add_batch(self, batch, results):
        """ Add a batch to calculate the metric on """

        yt = batch[self.target]
        yp = results[self.target]
        if len(yp.shape) > len(yt.shape):
            yt = yt.unsqueeze(-1).expand_as(yp)

        self.loss += self.loss_fn(yt, yp)
        self.n_entries += np.prod(yt.shape)

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        return self.loss / self.n_entries

    @staticmethod
    def loss_fn(yt, yp):
        """Calculates loss function for yt and yp"""
        raise NotImplementedError


class MeanSquaredError(Metric):
    r"""
    Metric for mean square error. For non-scalar quantities, the mean of all
    components is taken.

    Args:
        target (str): name of target property
        name (str): name used in logging for this metric. If set to `None`,
            `MSE_[target]` will be used (Default: None)
    """

    def __init__(
        self,
        target,
        name=None,
    ):
        name = "MSE_" + target if name is None else name
        super().__init__(
            target=target,
            name=name,
        )

    @staticmethod
    def loss_fn(yt, yp):
        yt, yp = yt.to(torch.float), yp.to(torch.float)
        if len(yp.shape) > len(yt.shape):
            yt = yt.unsqueeze(-1).expand_as(yp)
        diff = yt - yp
        return torch.sum(diff.view(-1) ** 2).detach().cpu().data.numpy()


class RootMeanSquaredError(MeanSquaredError):
    r"""
    Metric for root mean square error. For non-scalar quantities, the mean of
    all components is taken.

    Args:
        target (str): name of target property
        name (str): name used in logging for this metric. If set to `None`,
            `RMSE_[target]` will be used (Default: None)
    """

    def __init__(
        self,
        target,
        name=None,
    ):
        name = "RMSE_" + target if name is None else name
        super().__init__(
            target, name
        )

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        return np.sqrt(self.loss / self.n_entries)


class MeanAbsoluteError(Metric):
    r"""
    Metric for mean absolute error. For non-scalar quantities, the mean of all
    components is taken.

    Args:
        target (str): name of target property
        name (str): name used in logging for this metric. If set to `None`,
            `MAE_[target]` will be used (Default: None)
    """

    def __init__(
        self,
        target,
        name=None,
    ):
        name = "MAE_" + target if name is None else name
        super().__init__(
            target=target,
            name=name,
        )

    @staticmethod
    def loss_fn(yt, yp):
        yt, yp = yt.to(torch.float), yp.to(torch.float)
        if len(yp.shape) > len(yt.shape):
            yt = yt.unsqueeze(-1).expand_as(yp)
        diff = yt - yp
        return torch.sum(torch.abs(diff).view(-1)).detach().cpu().data.numpy()
