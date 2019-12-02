"""This file contains the utility class DataProvider for managing datasets.

It contains RecDataset which stores users, items, user-item interaction pairs and 
DataProvider which provides data for training and evaluation.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os

RATING_NAMES = ["user", "item", "rating"]

class RecDataset():
    """A dataset class storing users, items and user-item interactions.
    
    It loads training data and test data from the input directory path and 
    provides interfaces about information of dataset like the number of users 
    and items.
    
    Attributes:
    
    """
    def __init__(self, dir_path):
        """
        RecDataset construct datasets from user-item interaction pairs.
        """
        rating_train = pd.read_csv(os.path.join(dir_path,"movielens-100k-train.txt"), names = RATING_NAMES, sep='\t')
        rating_train = rating_train[rating_train.rating > 3.99]
        rating_test = pd.read_csv(os.path.join(dir_path,"movielens-100k-test.txt"), names = RATING_NAMES, sep='\t')
        rating_test = rating_test[rating_test.rating > 3.99]
        
        # Build ID transformer
        user_id_transformer = self._build_df_id_transformer(rating_train.append(rating_test),"user")
        item_id_transformer = self._build_df_id_transformer(rating_train.append(rating_test),"item")
        
        # Transform id to let ids of users and items start from 0
        rating_train.loc[:, 'user'] = rating_train['user'].map(lambda x:user_id_transformer[x])
        rating_train.loc[:, 'item'] = rating_train['item'].map(lambda x:item_id_transformer[x])
        rating_test.loc[:, 'user'] = rating_test['user'].map(lambda x:user_id_transformer[x])
        rating_test.loc[:, 'item'] = rating_test['item'].map(lambda x:item_id_transformer[x])
        
        self._rating_train = rating_train
        self._rating_test = rating_test
        
        self._users, self._items = self._extract_users_and_items(self._rating_train.append(self._rating_test))
        self._bought_dict = self.get_interaction_records()
    
    def _build_df_id_transformer(self, df, col):
        """ Builds a transformer for a column letting its values start from 0
        
        Args:
           df: DataFrame to be processed
           col: Column name
          
        Returns:
            A dict mapping old ids to new ids
        """
        values = df[col].unique()
        return dict(zip(values, range(len(values))))
    
    def _extract_users_and_items(self, rating):
        """Extract users and items from rating data
        
        Args:
            rating: rating records in pandas DataFrame.
           
        Returns:
            users: users stored in np.ndarray 
            items: items stored in np.ndarray
        """
        users = rating.user.unique()
        items = rating.item.unique()
        return users, items
    
    def get_interaction_records(self, method = "train"):
        '''Gets interaction records for each user
        
        Args:
            rating: rating records in pandas DataFrame.
            
        Returns:
            A dict mapping users to items interacted with him/her.
        '''
        if(method == "train"):
            return dict(self._rating_train.groupby("user")['item'].apply(np.array))
        else:
            return dict(self._rating_test.groupby("user")['item'].apply(np.array))
            
    
    def get_users(self):
        return self._users
    
    def get_num_users(self):
        return len(self._users)
    
    def get_items(self):
        return self._items
    
    def get_num_items(self):
        return len(self._items)
    
    def get_bought_mask(self):
        """ Get bought records as masked tensors.
        
        Args:
        
        Returns:
            A zero-one tensor with size [N,M] where N is the number of users and M is the number of items.
            If user i bought item j, the j_th position at the i_th row is 1, otherwise 0.
        """
        bought_mask = []
        for user in self._users:
            user = user.item()
            item_mask = torch.zeros(self.get_num_items())
            if user in self._bought_dict:
                item_mask[self._bought_dict[user]] = 1
            bought_mask.append(item_mask)
        return torch.stack(bought_mask)
    
    def get_user_item_pairs(self, method = "train"):
        """ Get user-item interaction pairs
        
        Args: 
            method: A string indicating to get training data or test data
       
        Returns:
            Users and items of interaction pairs.
        """
        if(method == "train"):
            return self._rating_train.user.to_numpy(), self._rating_train.item.to_numpy()
        else:
            return self._rating_test.user.to_numpy(), self._rating_test.item.to_numpy()
            
class DataProvider():
    """Provides training datasets
    """
    def __init__(self, device = "cpu"):
        self.device = device
    
    def prepare_data_for_generator(self, users, batch_size = 64):
        """Prepare training data for generator.
        
        Provides batches of users for generator.
        
        Args:
            users: Users to be split into batches.
            batch_size: Batch size with.
           
        Returns:
            A DataLoader which yields batches of users.
        """
        users = TensorDataset(torch.from_numpy(users).to(self.device))
        return DataLoader(users, batch_size = batch_size, shuffle=True)
    
    def prepare_data_for_discriminator(self, generator, real_ui_pairs, k = 128, temperature=0.2, lambda_bought=0, batch_size = 64):
        """Prepares training data for discriminator.
        
        Provides real user-item pairs mixed with fake user-item pairs in batches.
        
        Args:
            generator: Generator from which to sample fake user-item pairs.
            real_ui_pairs: Real user-item interaction pairs.
            k: The number of fake samples for each sampled user.
            temperature: Temperature for annealed softmax.
            lambda_bought: Lambda for importance sampling.
            batch_size: Batch size.
        
        Returns:
            A DataLoader which yields batches of (user, item, label) pairs.
        """   
        # Get Positive Data
        real_users, real_items = real_ui_pairs
        real_users = torch.tensor(real_users).to(self.device)
        real_items = torch.tensor(real_items).to(self.device)
        real_labels = torch.ones(len(real_users)).float().to(self.device)
    
        # Get Negative Data
        fake_users = real_users.unique()
        fake_items, _, _= generator.sample_items_for_users(fake_users, k = k, temperature = temperature, lambda_bought = lambda_bought)
        fake_users = fake_users.view(-1,1).expand_as(fake_items).contiguous()
        fake_users = fake_users.view(-1)
        fake_items = fake_items.view(-1)
        fake_labels = torch.zeros(len(fake_users)).float().to(self.device)
        
        users = torch.cat((real_users, fake_users), 0)
        items = torch.cat((real_items, fake_items), 0)
        labels = torch.cat((real_labels, fake_labels), 0)
        dataset = TensorDataset(users, items, labels)
        return DataLoader(dataset, batch_size = batch_size, shuffle=False)
    
    def prepare_bpr_triplets(self, items, bought_dict, batch_size = 64):
        """ Prepares triplets for models optimizing bpr-like loss.
        
        Provides triplets with the form (u, u_i, u_j) where u is the target user, u_i and u_j 
        are the positive item and negative item respectively.
        
        Args:
            items: a list of all items
            bought_dict: a dict mapping users to items they bought
            batch_size: Batch size.
        Returns:
            A DataLoader which yields batches of (u, u_i, u_j) triplets
        """
        train_users = []
        train_pos_items = []
        train_neg_items = []
        for user in bought_dict.keys():
            positive_items = bought_dict[user]
            candidate_neg_items = np.setdiff1d(items, positive_items)
            negative_items = np.random.choice(candidate_neg_items, len(positive_items))   
            train_user = np.array(user).repeat(len(positive_items))
            
            train_users.append(train_user)
            train_pos_items.append(positive_items)
            train_neg_items.append(negative_items)
        train_users = np.concatenate(train_users).flatten()
        train_pos_items = np.concatenate(train_pos_items).flatten()
        train_neg_items = np.concatenate(train_neg_items).flatten()
        train_users = torch.tensor(train_users).to(self.device)
        train_pos_items = torch.tensor(train_pos_items).to(self.device)
        train_neg_items = torch.tensor(train_neg_items).to(self.device)        
        dataset = TensorDataset(train_users,train_pos_items,train_neg_items)
        return DataLoader(dataset, batch_size = batch_size, shuffle=False)
    
    def prepare_bpr_triplets_dns(self, model, bought_mask, batch_size = 64, dns_k = 5):
        """Prepares bpr triplets with dynamic negative sampling(DNS) for pre-train models.
        
        Args:
            model: Model for dynamic negative sampling
            bougth_mask: A mask for recording users' historical interaction records on items
            dns_k: The number of candidate items for DNS
        Returns:
            A DataLoader which yields batches of (u, u_i, u_j) triplets
        
        """
        scores = model.get_scores_for_all_items()
        uniform_dis = torch.ones_like(scores).float()
        uniform_dis[bought_mask.bool()] = 0
        
        train_users = []
        train_pos_items = []
        train_neg_items = []
        for user, (sample_dis, rating) in enumerate(zip(uniform_dis, scores)):
            user_mask = bought_mask[user]
            pos_items = user_mask.nonzero().squeeze(1)
            num_pos_items = user_mask.sum().int()
            candidate_items = torch.multinomial(sample_dis, num_pos_items * dns_k, replacement = True)
            _, indices = torch.topk(rating[candidate_items], num_pos_items)
            neg_items = candidate_items[indices]
            user = torch.ones_like(pos_items) * user
            
            train_users.append(user)
            train_pos_items.append(pos_items)
            train_neg_items.append(neg_items)
        train_users = torch.cat(train_users).to(self.device)  
        train_pos_items = torch.cat(train_pos_items).to(self.device)  
        train_neg_items = torch.cat(train_neg_items).to(self.device)      
        dataset = TensorDataset(train_users,train_pos_items,train_neg_items)
        return DataLoader(dataset, batch_size = batch_size, shuffle=False)