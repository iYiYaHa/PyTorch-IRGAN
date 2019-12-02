""" This file contains tools for performance evaluation

It supports general and widely adopted leave-one-out evaluation protocol.
And it supports evaluation metrics like NDCG, MRR, MAP and hit ratio.
"""

from evaluation.rank_metrics import ndcg_at_k, mean_average_precision, mean_reciprocal_rank, precision_at_k
import numpy as np
from tqdm import tqdm 
import torch

class RecEvaluator(object):
    """Evaluator provides methods for metric evaluation
    """
    def __init__(self, test_set, neg_set = None, device="cpu"):
        """Initialize Evaluator with test set and negative samples set
        
        Args:
            test_set: A dict mapping users to their corresponding test set.
            neg_set: A dict mapping users to their corresponding negative samples set.(For leave-one-out evaluation)
        """
        self._predict_users, self._gt_items = list(zip(*test_set.items()))
        self._rank_items = [np.append(test_set[user], neg_set[user]) for user in self._predict_users] if neg_set != None else None
        self._device = device
    
    def top_k_evaluation(self, model, k_list = [5], batch_size = 64):
        """Evaluates the performance of model with top k evaluation protocol
        
        Args:
            model: Recommender model to be evaluated.
            k: The number of items to recommend for each user
            
        Returns:
            Ndcg, hit ratio and mean reciprocal rank results.
        """
        users = torch.tensor(self._predict_users).to(self._device)
        predicted_items = [model.top_k_items_for_users(batch_users, max(k_list)) for batch_users in _split_into_chunks(users, batch_size)]
        predicted_items = torch.cat(predicted_items).cpu().numpy()
        return [self._eval_metrics(predicted_items, self._gt_items, k) for k in k_list] 
    
    def leave_one_out_evaluation(self, model, k_list = [5], batch_size = 64):
        """Evaluates the performance of model with leave-one-out evaluation protocol
        
        Args:
            model: Recommender model to be evaluated.
            
        Returns:
            NDCG, hit ratio, mean average precision and mean reciprocal rank results.
        """
        assert self._rank_items != None, "Empty negative samples set."
        users = torch.tensor(self._predict_users).to(self._device)
        items = torch.tensor(self._rank_items).to(self._device)
        predicted_items = [model.rank_items_for_users(users[batch_users], items[batch_users]) \
                           for batch_users in _split_into_chunks(self._predict_users, batch_size)]
        predicted_items = torch.cat(predicted_items).cpu().numpy()
        return [self._eval_metrics(predicted_items, self._gt_items, k) for k in k_list] 
    
    def _eval_metrics(self, predicted_result, ground_truth, k = 5):
        """Evaluates predicted results on metrics including NDCG@k, precision@k, hit@k, map and mrr. 
        """
        ndcgK = self._eval_ndcg(predicted_result, ground_truth, k = k)
        precisionK = self._eval_precision(predicted_result, ground_truth, k = k)
        hitK = self._eval_hit_ratio(predicted_result, ground_truth, k = k)
        mapK = self._eval_map(predicted_result, ground_truth) 
        mrr = self._eval_mrr(predicted_result, ground_truth)
        return ndcgK, precisionK, hitK, mapK, mrr
    
    def _eval_ndcg(self, predicted_result, ground_truth, k = 5):
        assert len(predicted_result) == len(ground_truth),"Different size of predicted items and test items."
        ndcg = 0
        for items, ground_truth_items in zip(predicted_result, ground_truth):
            scores = np.zeros(len(items))
            for ind in range(len(items)):
                if items[ind] in ground_truth_items:
                    scores[ind] = 1
            ndcg += ndcg_at_k(scores, k)
        return ndcg / len(predicted_result)
    
    def _eval_hit_ratio(self, predicted_result, ground_truth, k = 5):
        assert len(predicted_result) == len(ground_truth),"Different size of predicted items and test items."
    
        hit_num = 0
        total_num = 0
        for items, ground_truth_items in zip(predicted_result, ground_truth):
            hit_num += len([item for item in items[:k] if item in ground_truth_items])
            total_num += len(ground_truth_items)
        return hit_num / total_num
    
    def _eval_precision(self, predicted_result, ground_truth, k = 5):
        assert len(predicted_result) == len(ground_truth),"Different size of predicted items and test items."
        precision = 0
        for items, ground_truth_items in zip(predicted_result, ground_truth):
            scores = np.zeros(len(items))
            for ind in range(len(items)):
                if items[ind] in ground_truth_items:
                    scores[ind] = 1
            precision += precision_at_k(scores, k)
        return precision / len(predicted_result)
        
    def _eval_mrr(self, predicted_result, ground_truth):
        scores = _convert_to_zero_one_vec(predicted_result, ground_truth)
        return mean_reciprocal_rank(scores)
    
    def _eval_map(self, predicted_result, ground_truth):
        assert len(predicted_result) == len(ground_truth),"Different size of predicted items and test items."
        res = _convert_to_zero_one_vec(predicted_result, ground_truth)
        return mean_average_precision(res)
    
####################################################
# Utility Function
####################################################

def _split_into_chunks(lst, n):
    """Splits a list into chunks with size of n
    
    Args:
        lst: List to be chunked
        n: The size of each chunk
        
    Yields:
        A chunk of the original list
    """
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _convert_to_zero_one_vec(pred_items, true_items):
    """Convert pred_items into zero-one arrays based whether each item appeared in true_items.
    
    Args:
        pred_items: List of items to be converted
        true_items: List of ground truth items
        
    Returns:
        An array of zero-one arrays.
    """
    scores = []
    for items,ground_truth_items in zip(pred_items,true_items):
        score = np.zeros(len(items))
        for ind in range(len(items)):
            if items[ind] in ground_truth_items:
                score[ind] = 1
        scores.append(score)
    return scores