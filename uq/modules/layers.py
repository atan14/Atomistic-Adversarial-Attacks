import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_
from functools import partial

from .construct import layer_types
from evi.utils.graphop import scatter_add


DEFAULT_DROPOUT_RATE = 0.0
zeros_initializer = partial(constant_, val=0.0)


class MessagePassingModule(nn.Module):

    """Convolution constructed as MessagePassing.
    """

    def __init__(self):
        super(MessagePassingModule, self).__init__()

    def message(self, r, e, a, aggr_wgt):
        # Basic message case
        assert r.shape[-1] == e.shape[-1]
        # mixing node and edge feature, multiply by default
        # possible options:
        # (ri [] eij) -> rj,
        # where []: *, +, (,), permutation....
        if aggr_wgt is not None:
            r = r * aggr_wgt

        message = r[a[:, 0]] * e, r[a[:, 1]] * e
        return message

    def aggregate(self, message, index, size):
        # pdb.set_trace()
        new_r = scatter_add(src=message,
                            index=index,
                            dim=0,
                            dim_size=size)
        return new_r

    def update(self, r):
        return r

    def forward(self, r, e, a, aggr_wgt=None):

        graph_size = r.shape[0]

        rij, rji = self.message(r, e, a, aggr_wgt)
        # i -> j propagate
        r = self.aggregate(rij, a[:, 1], graph_size)
        # j -> i propagate
        r += self.aggregate(rji, a[:, 0], graph_size)
        r = self.update(r)
        return r


def gaussian_smearing(distances, offset, widths, centered=False):

    if not centered:
        # Compute width of Gaussians (using an overlap of 1 STDDEV)
        # widths = offset[1] - offset[0]
        coeff = -0.5 / torch.pow(widths, 2)
        diff = distances - offset

    else:
        # If Gaussians are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # If centered Gaussians are requested, don't substract anything
        diff = distances

    # Compute and return Gaussians
    gauss = torch.exp(coeff * torch.pow(diff, 2))

    return gauss


class GaussianSmearing(nn.Module):
    """
    Wrapper class of gaussian_smearing function. Places a predefined number of Gaussian functions within the
    specified limits.

    sample struct dictionary:

        struct = {'start': 0.0, 'stop':5.0, 'n_gaussians': 32, 'centered': False, 'trainable': False}

    Args:
        start (float): Center of first Gaussian.
        stop (float): Center of last Gaussian.
        n_gaussians (int): Total number of Gaussian functions.
        centered (bool):  if this flag is chosen, Gaussians are centered at the origin and the
              offsets are used to provide their widths (used e.g. for angular functions).
              Default is False.
        trainable (bool): If set to True, widths and positions of Gaussians are adjusted during training. Default
              is False.
    """

    def __init__(self,
                 start,
                 stop,
                 n_gaussians,
                 centered=False,
                 trainable=False):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor(
            (offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Tensor of interatomic distances.

        Returns:
            torch.Tensor: Tensor of convolved distances.

        """
        result = gaussian_smearing(
            distances, self.offsets, self.width, centered=self.centered
        )

        return result


class Dense(nn.Linear):
    """ Applies a dense layer with activation: :math:`y = activation(Wx + b)`

    Args:
        in_features (int): number of input feature
        out_features (int): number of output features
        bias (bool): If set to False, the layer will not adapt the bias. (default: True)
        activation (callable): activation function (default: None)
        weight_init (callable): function that takes weight tensor and initializes (default: xavier)
        bias_init (callable): function that takes bias tensor and initializes (default: zeros initializer)
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        dropout_rate=DEFAULT_DROPOUT_RATE,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
    ):

        self.weight_init = weight_init
        self.bias_init = bias_init

        super().__init__(in_features, out_features, bias)

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def reset_parameters(self):
        """
            Reinitialize model parameters.
        """
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Output of the dense layer.
        """
        self.to(inputs.device)
        y = super().forward(inputs)

        if hasattr(self, "dropout"):
            y = self.dropout(y)

        if self.activation:
            y = self.activation(y)

        return y


class ScaleShift(nn.Module):

    r"""Scale and shift layer for standardization.
    .. math::
       y = x \times \sigma + \mu
    Args:
        means (dict): dictionary of mean values
        stddev (dict): dictionary of standard deviations
    """

    def __init__(self,
                 means=None,
                 stddevs=None):
        super(ScaleShift, self).__init__()

        means = means if (means is not None) else {}
        stddevs = stddevs if (stddevs is not None) else {}
        self.means = means
        self.stddevs = stddevs

    def forward(self, inp, key):
        """Compute layer output.
        Args:
            inp (torch.Tensor): input data.
        Returns:
            torch.Tensor: layer output.
        """

        stddev = self.stddevs.get(key, 1.0)
        mean = self.means.get(key, 0.0)
        out = inp * stddev + mean

        return out


def get_act(activation):
    return layer_types[activation]()


class Split(nn.Module):
    def __init__(self, output_key):
        super().__init__()
        self.output_key = output_key

    def forward(self, r):
        inds = {'v': 0, 'alpha': 1, 'beta': 2}
        return r[:, inds[self.output_key]]
