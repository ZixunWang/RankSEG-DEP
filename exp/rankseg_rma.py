import logging
from typing import Union

import torch
import numpy as np

TRUNCATE_PROB = 0.5
logger = logging.getLogger(__name__)


def convert_to_nonoverlap_prob(overlap_predict, prob, **kwargs):
    num_class = overlap_predict.size(0)
    overlap_mask = overlap_predict.sum(0) > 1
    nonoverlap_predict = torch.zeros_like(overlap_predict[0], dtype=torch.uint8)
    assert num_class <= 256, 'num_class should be less than 256, when using uint8'
    for c in range(num_class):
        safe_to_predict = overlap_predict[c] & ~overlap_mask
        nonoverlap_predict[safe_to_predict] = c
    argmax_mask = prob.argmax(0)
    nonoverlap_predict[overlap_mask] = argmax_mask[overlap_mask].type(torch.uint8)
    return nonoverlap_predict


def convert_to_nonoverlap_weight_prob(overlap_predict, prob, **kwargs):
    num_class = overlap_predict.size(0)
    overlap_mask = overlap_predict.sum(0) > 1
    nonoverlap_predict = torch.zeros_like(overlap_predict[0], dtype=torch.uint8)
    assert num_class <= 256, 'num_class should be less than 256, when using uint8'
    weight_prob = prob
    for c in range(num_class):
        safe_to_predict = overlap_predict[c] & ~overlap_mask
        nonoverlap_predict[safe_to_predict] = c
        volume = safe_to_predict.sum().item()
        weight_prob[c] /= 1 + volume
    argmax_mask = weight_prob.argmax(0)
    nonoverlap_predict[overlap_mask] = argmax_mask[overlap_mask].type(torch.uint8)
    return nonoverlap_predict

def convert_to_nonoverlap_rma_score_batched(overlap_predict, prob, opt_metric, sorted_prob, pb_mean, **kwargs):
    # Shapes:
    # overlap_predict, prob: (B, C, N)
    # sorted_prob: (B, C, K) - checking sorted_prob[:, :, 0]
    # pb_mean: (B, C)
    B, C, N = prob.shape
    device = prob.device
    
    # 1. Global Overlap Mask (B, N)
    # Identifies pixels where more than one class is predicted
    class_counts = overlap_predict.sum(dim=1)
    single_pred_mask = (class_counts == 1)
    overlap_mask = class_counts > 1

    # 2. Pruning Mask (B, C)
    # Identify which (batch, class) pairs are valid
    active_mask = sorted_prob[:, :, 0] > TRUNCATE_PROB

    # 3. Calculate 'Safe to Predict' for the whole batch (B, C, N)
    # overlap_mask (B, N) -> (B, 1, N) for broadcasting
    safe_to_predict = overlap_predict & ~overlap_mask.unsqueeze(1)

    # 4. Calculate mu and opt_tau (B, C, 1)
    # We sum across the pixel dimension (dim=2)
    mu = (prob * safe_to_predict.float()).sum(dim=2, keepdim=True)
    opt_tau = safe_to_predict.float().sum(dim=2, keepdim=True)

    # 5. Prepare increment_score (B, C, N)
    # Initialize with a very small number for argmax safety
    increment_score = torch.full_like(prob, float('-inf'))
    
    pb_mean_expanded = pb_mean.unsqueeze(2) # (B, C, 1)
    
    if opt_metric == 'dice':
        denom = opt_tau + pb_mean_expanded + 1
        scores = 2 * ((mu + prob) / (denom + 1) - mu / denom)
    else:
        denom = opt_tau + pb_mean_expanded - mu
        scores = (mu + prob) / (denom - prob + 1) - mu / denom

    # Apply pruning: only keep scores where active_mask is True
    # active_mask (B, C) -> (B, C, 1) for broadcasting
    increment_score = torch.where(active_mask.unsqueeze(2), scores, increment_score)
    
    # 6. Global Argmax across the Class dimension (dim=1)
    # Resulting shape: (B, N)
    increment_argmax_mask = increment_score.argmax(dim=1).to(torch.uint8)

    # 7. Final Output Construction
    # We only apply the increment_argmax where there was an overlap.
    # Otherwise, you might want a default class or the existing single prediction.
    # Based on your logic:
    nonoverlap_predict = torch.zeros((B, N), dtype=torch.uint8, device=device)
    nonoverlap_predict[single_pred_mask] = overlap_predict.to(torch.uint8).argmax(dim=1).to(torch.uint8)[single_pred_mask]
    nonoverlap_predict[overlap_mask] = increment_argmax_mask[overlap_mask]
    return nonoverlap_predict

def convert_to_nonoverlap_rma_score(overlap_predict, prob, opt_metric, sorted_prob, opt_tau, pb_mean, **kwargs):
    num_class = overlap_predict.size(0)
    nonoverlap_predict = torch.zeros_like(overlap_predict[0], dtype=torch.uint8)
    assert num_class <= 256, 'num_class should be less than 256, when using uint8'
    overlap_mask = overlap_predict.sum(0) > 1
    dim = overlap_predict.size(1)
    upper_bound_scale = (dim + 1) / dim
    increment_score = torch.zeros_like(prob, dtype=torch.float32)

    # active_classes = (sorted_prob[:, 0] > TRUNCATE_PROB).nonzero(as_tuple=True)[0]

    # pruned_prob = prob[active_classes]
    # pruned_overlap_predict = overlap_predict[active_classes]
    # pruned_pb_mean = pb_mean[active_classes]
    
    # safe_to_predict = pruned_overlap_predict & ~overlap_mask.unsqueeze(0)
    # mu = (pruned_prob * safe_to_predict).sum(dim=1, keepdim=True)
    # opt_tau = safe_to_predict.sum(dim=1, keepdim=True)
    # pruned_pb_mean_expanded = pruned_pb_mean.unsqueeze(1)

    # if opt_metric == 'dice':
    #     denom = opt_tau + pruned_pb_mean_expanded + 1
    #     increment_score[active_classes] = 2 * ((mu + pruned_prob) / (denom + 1) - mu / denom)
    # else:
    #     denom = opt_tau + pruned_pb_mean_expanded - mu
    #     increment_score[active_classes] = (mu + pruned_prob) / (denom - pruned_prob + 1) - mu / denom

    for c in range(num_class):
        if sorted_prob[c][0] <= TRUNCATE_PROB:  # TODO: review this prune
            continue
        safe_to_predict = overlap_predict[c] & ~overlap_mask
        nonoverlap_predict[safe_to_predict] = c
        mu = prob[c][safe_to_predict].sum()#.item()
        opt_tau_this_c = safe_to_predict.sum()#.item()
        if opt_metric == 'dice':
            increment_score[c] = 2 * ((mu + prob[c]) / (opt_tau_this_c + pb_mean[c] + 2) - mu / (opt_tau_this_c + pb_mean[c] + 1))  # lower bound
            # increment_score[c] = 2 * ((mu + prob[c]) / (opt_tau[c] + upper_bound_scale*pb_mean[c] + 1) - mu / (opt_tau[c] + upper_bound_scale*pb_mean[c]))  # upper bound
        else:
            increment_score[c] = (mu + prob[c]) / (opt_tau_this_c + pb_mean[c] - mu - prob[c] + 1) - mu / (opt_tau_this_c + pb_mean[c] - mu)  # lower bound
            # increment_score[c] = (mu + prob[c]) / (opt_tau[c] + upper_bound_scale*pb_mean[c] - mu - prob[c] + 1) - mu / (opt_tau[c] + upper_bound_scale*pb_mean[c] - mu)  # upper bound
    increment_argmax_mask = increment_score.argmax(0)

    nonoverlap_predict[overlap_mask] = increment_argmax_mask[overlap_mask].type(torch.uint8)
    return nonoverlap_predict


def rankseg_rma(
        prob: Union[torch.Tensor, np.ndarray],
        opt_metric: str='dice',
        allow_overlap: bool=False,
        in_batch: bool=True,
        return_tau: bool=False,
        to_nonverlap_method: str='rma_score',
    ) -> Union[torch.Tensor, tuple]:

    is_binary = (prob.shape[1] == 2) and not allow_overlap

    # Check input and convert to tensor if needed
    if isinstance(prob, np.ndarray):
        prob = torch.from_numpy(prob)
    prob = prob.float()

    # check if required memory too large (~3GB), then use cpu
    if prob.nelement() * prob.element_size() > 3e9:  # TODO: review this threshold
        prob = prob.cpu()
        logger.warning(f'Input tensor is too large: {prob.nelement() * prob.element_size() / 1e9:.2f} GB; use CPU instead')

    batch_size = prob.shape[0] if in_batch else 1
    prob = prob.unsqueeze(0) if not in_batch else prob

    if is_binary:
        prob = prob[:, 1:2, ...]
        num_class = 1

    assert opt_metric in ['iou', 'dice'], 'opt_metric should be iou or dice'

    if to_nonverlap_method == 'prob':
        to_nonverlap_fn = convert_to_nonoverlap_prob
    elif to_nonverlap_method == 'weight_prob':
        to_nonverlap_fn = convert_to_nonoverlap_weight_prob
    elif to_nonverlap_method == 'rma_score':
        to_nonverlap_fn = convert_to_nonoverlap_rma_score
    else:
        raise ValueError(f'Invalid to_nonverlap_method: {to_nonverlap_method}')
    
    device = prob.device
    num_class = prob.shape[1]
    img_shape = prob.shape[2:]

    prob = torch.flatten(prob, start_dim=2, end_dim=-1)
    dim = prob.shape[-1]

    predict = torch.zeros(batch_size, num_class, dim, dtype=torch.bool, device=device)

    sorted_prob, top_index = torch.sort(prob, dim=-1, descending=True)
    pb_mean = prob.sum(dim=-1)
    cumsum_prob = torch.cumsum(sorted_prob, dim=-1)

    # Compute optimal tau and cutpoint
    opt_tau = compute_opt_tau(opt_metric, pb_mean, cumsum_prob, dim, device)
    # logger.info(f"opt_tau: {opt_tau}, cutpoint: {cutpoint}")

    for b in range(batch_size):
        for c in range(num_class):
            if sorted_prob[b, c, 0] <= TRUNCATE_PROB:  # TODO: review this prune
                continue
            predict[b, c, top_index[b, c, :opt_tau[b, c]]] = True

    if allow_overlap:
        predict = predict.reshape(batch_size, num_class, *img_shape) if in_batch else predict.squeeze(0).reshape(num_class, *img_shape)
    else:
        if is_binary:
            nonoverlap_predict = predict[:, 0, ...].long()
        else:
            nonoverlap_predict = convert_to_nonoverlap_rma_score_batched(predict, prob, opt_metric, sorted_prob, pb_mean)
            # nonoverlap_predict = torch.zeros(batch_size, dim, dtype=torch.uint8, device=device)
            # for b in range(batch_size):
            #     nonoverlap_predict[b] = to_nonverlap_fn(predict[b], prob=prob[b], opt_metric=opt_metric, sorted_prob=sorted_prob[b], top_index=top_index[b], pb_mean=pb_mean[b], opt_tau=opt_tau[b])
        predict = nonoverlap_predict.reshape(batch_size, *img_shape) if in_batch else nonoverlap_predict.reshape(*img_shape)

    if return_tau:
        return predict, opt_tau
    return predict


def compute_opt_tau(
        opt_metric: str, 
        pb_mean: torch.Tensor,
        cumsum_prob: torch.Tensor,
        dim: int,
        device: torch.device
    ):
    """Compute optimal tau and cutpoint based on the selected metric."""
    if opt_metric == 'dice':
        discount = pb_mean.unsqueeze(-1) + torch.arange(1, dim + 1, device=device).view(1, 1, -1) + 1.0  # lower bound
        # discount = (dim+1)/dim * pb_mean.unsqueeze(-1) + torch.arange(1, dim + 1, device=device).view(1, 1, -1)  # upper bound
        metric_values = 2.0 * cumsum_prob / discount
    else:  # IoU metric
        discount = pb_mean.unsqueeze(-1) - cumsum_prob + torch.arange(1, dim + 1, device=device).view(1, 1, -1)  # lower bound
        # discount = (dim+1)/dim * (pb_mean.unsqueeze(-1) - cumsum_prob) + torch.arange(1, dim + 1, device=device).view(1, 1, -1) - 1.0  # upper bound
        metric_values = cumsum_prob / discount

    # Get optimal tau indices
    opt_tau = torch.argmax(metric_values, dim=-1) + 1
    # cutpoint = sorted_prob[torch.arange(batch_size)[:, None], torch.arange(num_class), opt_tau - 1]

    return opt_tau

def rankseg_rma_(
        probs: torch.Tensor,
        metric: str="dice",
        smooth: float=0.0,
        output_mode: str='multiclass',
        pruning_prob: float=0.5,
        **kwargs
    ) -> torch.Tensor:
    """
    Produce the predicted segmentation by `rankdice` based on the estimated output probability.

    Parameters
    ----------
    probs : Tensor, shape (batch_size, num_class, \*image_shape)
        The estimated probability tensor.

    metric : str, default='dice'
        The metric aim to optimize, either 'iou' or 'dice'.

    output_mode : {'multiclass', 'multilabel'}, default='multiclass'
        Controls overlap behavior of the predictions.
        - 'multiclass': non-overlapping; each pixel belongs to exactly one class.
        - 'multilabel': overlapping; pixels can belong to multiple classes (binary mask per class).

    smooth : float, default=0.0
        A smooth parameter in the Dice metric.

    pruning_prob : float, default=0.5
        The threshold for pruning, if all probabilities are less than `pruning_prob`, 
        we skip the class.

    Returns
    -------
    preds : Tensor
        Shape (batch_size, num_class, \*image_shape) if output_mode == 'multilabel',
        otherwise shape (batch_size, \*image_shape)

    References
    ----------
    :cite:p:`wang2025rankseg` Wang, Z., & Dai, B. (2025). RankSEG-RMA: An Efficient Segmentation Algorithm via Reciprocal Moment Approximation. arXiv preprint arXiv:2510.15362.
    """

    assert metric in ['iou', 'dice'], 'metric should be iou or dice'

    def compute_opt_tau(
            metric: str,
            pb_mean: torch.Tensor,
            cumsum_prob: torch.Tensor,
            dim: int,
            smooth: float,
        ):
        """Compute optimal tau and cutpoint based on the selected metric."""
        device = pb_mean.device
        taus = torch.arange(1, dim + 1, device=device).view(1, 1, -1)
        if metric == 'dice':
            discount = pb_mean.unsqueeze(-1) + taus + 1.0 + smooth
            metric_values = 2.0 * cumsum_prob / discount
            metric_values += smooth / (discount - 1)
        elif metric == 'iou':
            discount = pb_mean.unsqueeze(-1) - cumsum_prob + taus + smooth
            metric_values = (cumsum_prob + smooth) / discount
        else:
            raise ValueError(f'Unsupported metric: {metric}')

        # Get optimal tau indices
        opt_tau = torch.argmax(metric_values, dim=-1) + 1
        # cutpoint = sorted_prob[torch.arange(batch_size)[:, None], torch.arange(num_class), opt_tau - 1]
        return opt_tau

    def convert_to_nonoverlap(
            overlap_preds: torch.Tensor,
            probs: torch.Tensor,
            metric: str,
            sorted_prob: torch.Tensor,
            pb_mean: torch.Tensor,
            smooth: float,
            pruning_prob: float,
        ) -> torch.Tensor:
        batch_size, num_classes, dim = probs.size()
        
        class_counts = overlap_preds.sum(dim=1)
        single_pred_mask = (class_counts == 1)
        overlap_mask = class_counts > 1
        safe_to_predict = overlap_preds & ~overlap_mask.unsqueeze(1)

        mu = (probs * safe_to_predict.float()).sum(dim=2, keepdim=True)
        opt_tau = safe_to_predict.float().sum(dim=2, keepdim=True)

        active_mask = sorted_prob[:, :, 0] > pruning_prob
        
        if metric == 'dice':
            denom = opt_tau + pb_mean.unsqueeze(2) + 1 + smooth
            increment_scores = 2 * ((mu + probs + smooth) / (denom + 1) - (mu + smooth) / denom)
        else:
            denom = opt_tau + pb_mean.unsqueeze(2) - mu + smooth
            increment_scores = (mu + probs + smooth) / (denom - probs + 1) - (mu + smooth) / denom
        
        increment_score = torch.where(active_mask.unsqueeze(2), increment_scores, torch.full_like(increment_scores, float('-inf')))

        increment_argmax_mask = increment_score.argmax(dim=1)
        nonoverlap_predict = torch.zeros((batch_size, dim), dtype=torch.int16, device=device)
        nonoverlap_predict[single_pred_mask] = overlap_preds.to(torch.int16).argmax(dim=1).to(torch.int16)[single_pred_mask]
        nonoverlap_predict[overlap_mask] = increment_argmax_mask[overlap_mask].to(torch.int16)

        return nonoverlap_predict

    return_binary_masks = (output_mode == 'multilabel')
    is_binary = (probs.shape[1] == 2) and not return_binary_masks

    if is_binary:
        probs = probs[:, 1:2, ...]
        num_classes = 1

    device = probs.device
    batch_size, num_classes, *image_shape = probs.shape

    probs = torch.flatten(probs, start_dim=2, end_dim=-1)
    dim = probs.shape[-1]

    sorted_prob, top_index = torch.sort(probs, dim=-1, descending=True)
    pb_mean = probs.sum(dim=-1)
    cumsum_prob = torch.cumsum(sorted_prob, dim=-1)

    overlap_preds = torch.zeros(batch_size, num_classes, dim, dtype=torch.bool, device=device)
    opt_tau = compute_opt_tau(metric, pb_mean, cumsum_prob, dim, smooth)
    for b in range(batch_size):
        for c in range(num_classes):
            if sorted_prob[b, c, 0] <= pruning_prob:  # TODO: review this prune
                continue
            overlap_preds[b, c, top_index[b, c, :opt_tau[b, c]]] = True

    if return_binary_masks:
        preds = overlap_preds.reshape(batch_size, num_classes, *image_shape)
    else:
        if is_binary:
            nonoverlap_preds = overlap_preds[:, 0, ...].long()
        else:
            nonoverlap_preds = convert_to_nonoverlap(
                overlap_preds,
                probs,
                metric,
                sorted_prob,
                pb_mean,
                smooth,
                pruning_prob,
            )
        preds = nonoverlap_preds.reshape(batch_size, *image_shape)

    return preds.long()
