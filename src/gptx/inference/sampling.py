"""Sampling methods for autoregressive decoding."""

import torch


def top_p(
    logits: torch.Tensor, p: float = 1.0, temperature: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-p (nucleus) sampling

    Only keep a subset of most probable tokens of which the cumulative probability is greater than p.
    Mask the rest of the logits with -inf.

    Parameters
    ----------
    logits : torch.Tensor
        shape (..., vocab_size)
    p : float
        Threshold of the cumulative probability of the top tokens, default is 1.0
    temperature : float
        Temperature for softmax; default is 1.0

    Returns
    -------
    logits_filtered : torch.Tensor
        logits masked with -inf for tokens below the threshold; shape (..., vocab_size)
    probs : torch.Tensor
        probabilities of the top p tokens renormalized; shape (..., vocab_size)

    Examples
    --------
    >>> top_p(logits=torch.tensor([[0., 1., 0., 1.]]), p=0.5)
    (tensor([[-inf, 1., -inf, 1.]]), tensor([[0.0000, 0.5000, 0.0000, 0.5000]]))
    >>> top_p(logits=torch.tensor([[1., 0., 1., 0.]]), p=0.5)
    (tensor([[1., -inf, 1., -inf]]), tensor([[0.5000, 0.0000, 0.5000, 0.0000]]))
    >>> top_p(logits=torch.tensor([[1., 1., 1., 1.]]), p=0.9999)
    (tensor([[1., 1., 1., 1.]]), tensor([[0.2500, 0.2500, 0.2500, 0.2500]]))
    >>> top_p(logits=torch.tensor([[[0.0, 0.1], [1.0, 0.9]]]), p=0.0001)
    (tensor([[[  -inf, 0.1000], [1.0000,   -inf]]]),
     tensor([[[0., 1.], [1., 0.]]]))
    """
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must be in the range (0.0, 1.0] but is {p=}")
    probs = torch.softmax(logits / temperature, dim=-1)  # shape (..., vocab_size)
    probs_sorted, indices_to_sorted = torch.sort(
        probs, descending=True, dim=-1
    )  # shape (..., vocab_size)
    probs_cum_sum = probs_sorted.cumsum(dim=-1)  # shape (..., vocab_size)
    mask_sorted_should_remove: torch.Tensor = (
        probs_cum_sum >= p
    )  # shape (..., vocab_size)
    # shift the mask to the right by 1 to keep the first token that exceeds the threshold as well
    mask_sorted_should_remove = mask_sorted_should_remove.roll(shifts=1, dims=-1)
    mask_sorted_should_remove[..., 0] = False  # shape (..., vocab_size)
    # mask the logits with -inf
    mask_should_remove = torch.empty_like(
        mask_sorted_should_remove
    )  # shape (..., vocab_size)
    mask_should_remove.scatter_(
        dim=-1, index=indices_to_sorted, src=mask_sorted_should_remove
    )
    logits.masked_fill_(mask_should_remove, -torch.inf)
    probs.masked_fill_(mask_should_remove, 0.0)
    probs /= probs.sum(dim=-1, keepdim=True)
    return logits, probs


if __name__ == "__main__":
    import doctest

    doctest.testmod()
