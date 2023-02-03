import torch
from torch.autograd import grad
from itertools import repeat


def compute_grad(inputs, output, allow_unused=False):
    """Compute gradient of the scalar output with respect to inputs.

    Args:
        inputs (torch.Tensor): torch tensor, requires_grad=True
        output (torch.Tensor): scalar output

    Returns:
        torch.Tensor: gradients with respect to each input component
    """

    assert inputs.requires_grad

    (gradspred,) = grad(
        output,
        inputs,
        grad_outputs=output.data.new(output.shape).fill_(1),
        create_graph=True,
        retain_graph=True,
        allow_unused=allow_unused,
    )

    return gradspred


def gen(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    dim = range(src.dim())[dim]  # Get real dim value.

    # Automatically expand index tensor to the right dimensions.
    if index.dim() == 1:
        index_size = list(repeat(1, src.dim()))
        index_size[dim] = src.size(dim)
        index = index.view(index_size).expand_as(src)

    # Generate output tensor if not given.
    if out is None:
        dim_size = index.max().item() + 1 if dim_size is None else dim_size
        out_size = list(src.size())
        out_size[dim] = dim_size
        out = src.new_full(out_size, fill_value)

    return src, out, index, dim


def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):

    src, out, index, dim = gen(
        src=src, index=index, dim=dim, out=out, dim_size=dim_size, fill_value=fill_value
    )
    output = out.scatter_add_(dim, index, src)

    return output


def split_and_sum(tensor, N):
    """spliting a torch Tensor into a list of uneven sized tensors,
    and sum each tensor and stack

    Example:
        A = torch.rand(10, 10)
        N = [4,6]
        split_and_sum(A, N).shape # (2, 10)

    Args:
        tensor (torch.Tensor): tensors to be split and summed
        N (list): list of number of atoms

    Returns:
        torch.Tensor: stacked tensor of summed smaller tensor
    """
    batched_prop = list(torch.split(tensor, N))

    for batch_idx in range(len(N)):
        batched_prop[batch_idx] = torch.sum(batched_prop[batch_idx], dim=0)

    return torch.stack(batched_prop)


def batch_and_sum(dict_input, N, predict_keys, xyz):
    """
    Pooling function to get graph property.
    Separate the outputs back into batches, pool the results,
    compute gradient of scalar properties if "_grad" is in the key name.
    same as SumPool class in evi.modules.schnet

    Args:
        dict_input (dict): Description
        N (list): number of batches
        predict_keys (list): Description
        xyz (tensor): xyz of the molecule

    Returns:
        dict: batched and pooled results
    """

    results = dict()

    for key, val in dict_input.items():
        # split
        if key in predict_keys and key + "_grad" not in predict_keys:
            results[key] = split_and_sum(val, N)
        elif key in predict_keys and key + "_grad" in predict_keys:
            results[key] = split_and_sum(val, N)
            grad = compute_grad(inputs=xyz, output=results[key])
            results[key + "_grad"] = grad
        # For the case only predicting gradient
        elif key not in predict_keys and key + "_grad" in predict_keys:
            results[key] = split_and_sum(val, N)
            grad = compute_grad(inputs=xyz, output=results[key])
            results[key + "_grad"] = grad

    return results


def make_directed(nbr_list):

    gtr_ij = (nbr_list[:, 0] > nbr_list[:, 1]).any().item()
    gtr_ji = (nbr_list[:, 1] > nbr_list[:, 0]).any().item()
    directed = gtr_ij and gtr_ji

    if directed:
        return nbr_list, directed

    new_nbrs = torch.cat([nbr_list, nbr_list.flip(1)], dim=0)
    return new_nbrs, directed
