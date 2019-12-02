'''
    This file contains model definition of the discriminator and generator for IRGAN.
'''

__author__ = "Youjie Zhang"

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """Generator for IRGAN
    
    Note that the user of generator have to make sure that IDs of users 
    and items start from 0.
    """
    def __init__(self, num_users, num_items, emb_dim, bought_mask):
        """Initialize embedding layer and bought_mask
        
        Args:
            num_users: The number of users.
            num_items: The number of items.
            emb_dim: Embedding dimension of embedding layer.
            bought_mask: Masked tensor for users representing their interaction records.
        """
        super(Generator, self).__init__()
        self.emb_users = nn.Embedding(num_users, emb_dim)
        self.emb_items = nn.Embedding(num_items, emb_dim)
        self.bias_items = nn.Embedding(num_items, 1)
        nn.init.uniform_(self.emb_users.weight, -0.05, 0.05)
        nn.init.uniform_(self.emb_items.weight, -0.05, 0.05)
        nn.init.zeros_(self.bias_items.weight)
        self.bought_mask = bought_mask
    
    def get_scores_for_all_items(self, users):
        """Get scores of items for some users.
        
        Args:
            users: Target users.
            
        Returns:
            A tensor containing scores of all items for some users.
        """
        selected_emb_users = self.emb_users(users)
        all_emb_items = self.emb_items.weight
        all_bias_items = self.bias_items.weight
        scores = torch.matmul(selected_emb_users, all_emb_items.t()) + all_bias_items.view(1,-1) # (#users, #items)
        return scores
    
    def get_scores_for_user_item_pairs(self, users, items):
        """Get scores for some user item pairs.
        
        Args:
            users: Target users.
            items: Target items.
            
        Returns:
            A tensor containing scores for user-item pairs.
        """
        selected_emb_users = self.emb_users(users)
        selected_emb_items = self.emb_items(items)
        selected_bias_items = self.bias_items(items)
        scores = torch.sum(torch.mul(selected_emb_users, selected_emb_items), 1) + selected_bias_items
        return scores
    
    def annealed_softmax(self, scores, temperature = 1):
        """Apply annealed softmax on scores to obtain probabilities
        
        Args:
            scores: Scores to apply softmax on.
            temperature: Temperature.
            
        Returns:
            Probabilities by applying annealed softmax on scores.
        """
        annealed_scores = scores / temperature
        probs = torch.softmax(annealed_scores, dim=1)
        return probs
    
    def sample_items_for_users(self, users, k = 5, temperature = 1, lambda_bought = 0):
        """Sample items for each user with annealed softmax and importance sampling.
        
        Args:
            users: Users to sample items for.
            k: The numer of items to be sampled for each user.
            temperature: Temperature for annealed softmax
            lambda_bought: Lambda for importance sampling
            
        Returns:
            A tensor containing k sampled items for each user.
        """
        scores = self.get_scores_for_all_items(users)
        probs = self.annealed_softmax(scores, temperature)
        p_n = (1 - lambda_bought) * probs
        p_n += lambda_bought * self.bought_mask[users] * 1.0/torch.sum(self.bought_mask[users],1).view(-1,1)
        
        sampled_items = torch.multinomial(p_n, k, replacement=True)
        sampled_items_probs = torch.gather(probs, dim = 1, index = sampled_items)
        sampled_items_p_n = torch.gather(p_n, dim = 1, index = sampled_items)
        return sampled_items, sampled_items_probs, sampled_items_p_n
    
    def top_k_items_for_users(self, users, k = 5, delete_bought = True):
        """ Gets top k items for users.
        
        Args:
            users: Target users.
            k: The number of items to recommend for each user.
            delete_bought: Boolean indicating whether recommend items bought before
        
        Returns:
            A tensor containing top k items for target users.
        """
        scores = self.get_scores_for_all_items(users)
        if delete_bought:# For items bought by each user before, assign values smaller than the current minimum score 
                    #to avoid them being chosen.
            scores[self.bought_mask[users].bool()] = scores.min() - 1
        _, top_k_items = torch.topk(scores, k)
        return top_k_items
    
    def rank_items_for_users(self, users, items):
        """Ranks candidate item set for each user according to their scores.
        
        Args:
            users: N users.
            items: N * M items representing M items for each user.
        
        Returns:
            Ranked items for each user.
        """
        N, M = items.size()
        expanded_users = users.view(-1, 1).expand_as(items)
        scores = self.get_scores_for_user_item_pairs(expanded_users.view(-1), items.view(-1))
        scores = scores.view(N,M)
        
        ranked_scores, ranked_indices = torch.sort(scores, descending = True)
        ranked_items = torch.gather(items, dim = 1, index = ranked_indices)
        return ranked_items
    
    def forward(self, users, k = 5):
        """Recommend top k items for each user
        
        Args:
            users: Users to recommend items for.
            k: The number of items to be recommended for each user.
            
        Returns:
            Top k items for each user.
        """
        return self.top_k_items_for_users(users, k)
        
        
class Discriminator(nn.Module):
    """Discriminator for IRGAN
    
    Note that methods top_k_items_for_users and get_scores_for_all_items are used for pre-train
    """
    def __init__(self, num_users, num_items, emb_dim, bought_mask = None):
        """Initialize embedding layers for users and items
        
        Args:
            num_users: The number of users.
            num_items: The number of items.
            emb_dim: Embedding dimension of embedding layer.
        """
        super(Discriminator, self).__init__()
        self.emb_users = nn.Embedding(num_users, emb_dim)
        self.emb_items = nn.Embedding(num_items, emb_dim)
        self.bias_items = nn.Embedding(num_items, 1)
        self.bought_mask = bought_mask
        nn.init.uniform_(self.emb_users.weight, -0.05, 0.05)
        nn.init.uniform_(self.emb_items.weight, -0.05, 0.05)
        nn.init.zeros_(self.bias_items.weight)    
        
    def top_k_items_for_users(self, users, k = 5, delete_bought = True):
        """ Gets top k items for users.
        
        Args:
            users: Target users.
            k: The number of items to recommend for each user.
            delete_bought: Boolean indicating whether recommend items bought before
        
        Returns:
            A tensor containing top k items for target users.
        """
        scores = self.get_scores_for_all_items(users)
        if delete_bought and self.bought_mask is not None:
            scores[self.bought_mask[users].bool()] = scores.min()-1
        _, top_k_items = torch.topk(scores, k)
        return top_k_items
        
    def get_scores_for_all_items(self, users = None):
        """Get scores of items for some users.
        
        Args:
            users: Target users.
            
        Returns:
            A tensor containing scores of all items for some users.
        """
        selected_emb_users = self.emb_users(users) if users is not None else self.emb_users.weight
        all_emb_items = self.emb_items.weight
        all_bias_items = self.bias_items.weight
        scores = torch.matmul(selected_emb_users, all_emb_items.t()) + all_bias_items.view(1,-1) # (#users, #items)
        return scores
    
    def forward(self, users, items):
        """ Get scores for user-item pairs.
        
        Args:
           users: Target users.
           items: Target items.
           
        Returns:
           A tensor containing scores for user-item pairs
        """
        selected_emb_users = self.emb_users(users)
        selected_emb_items = self.emb_items(items)
        selected_bias_items = self.bias_items(items).squeeze()
        scores = torch.sum(torch.mul(selected_emb_users, selected_emb_items), 1) + selected_bias_items
        return torch.sigmoid(scores)