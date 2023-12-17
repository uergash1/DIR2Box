import torch
import numpy as np
import pandas as pd
import pickle as pkl
import os
import random
import json
from tqdm import tqdm, trange
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import to_undirected


class Dataset:
    def __init__(self, args, device):
        super(Dataset, self).__init__()

        self.args = args
        self.device = device
        self.dataset_name = args.dataset
        self.__load_query_resource_artifacts__()

    def __load_query_resource_artifacts__(self):
        self.resource_query_similarity = pd.read_csv(
            f'../data/{self.dataset_name}/processed/resource_query_similarity_v{self.args.version}.tsv', sep='\t')

        with open(f'../data/{self.dataset_name}/processed/id_to_rname.json') as f:
            self.id_to_rname = json.load(f)

        # Loading queries
        self.query_ids = sorted(list(self.resource_query_similarity['query_id'].unique()))
        self.queries = np.load(f"../data/{self.dataset_name}/embeddings/{self.args.llm}/queries.npy")

        random.seed(2023)
        self.query_ids_cv = list(self.query_ids)
        random.shuffle(self.query_ids_cv)

        # Loading resources
        self.resource_ids = sorted(list(self.resource_query_similarity['resource_id'].unique()))

        self.documents = {}
        for resource_id in self.resource_ids:
            self.documents[resource_id] = np.load(
                f"../data/{self.dataset_name}/embeddings/{self.args.llm}/resources/{self.args.num_doc}/{self.id_to_rname[str(resource_id)]}.npy")

        self.resource_document_embedding = torch.Tensor(
            [self.documents[resource_id] for resource_id in self.resource_ids])

        # Resource-resource graph
        self.construct_resource_graph()

        # Query's pos/neg resources
        self.get_positive_negative_resources()

    def construct_resource_graph(self):
        graph_file = f'graphs/{self.args.dataset}_threshold{self.args.threshold}_doc{self.args.num_doc}.npy'

        if not os.path.exists(graph_file):
            self.edge_index, self.edge_weight = [], []
            for resource_i in trange(len(self.resource_ids)):
                for resource_j in range(resource_i + 1, len(self.resource_ids)):
                    sim = cosine_similarity(self.documents[resource_i], self.documents[resource_j])
                    sim_ij = np.sum(sim > self.args.threshold) / (sim.shape[0] * sim.shape[1])
                    if sim_ij > 0:
                        self.edge_index.append([resource_i, resource_j])
                        self.edge_weight.append(sim_ij)

            self.edge_index = torch.tensor(self.edge_index).T
            self.edge_weight = torch.tensor(self.edge_weight)
            self.edge_index, self.edge_weight = to_undirected(self.edge_index, self.edge_weight)

            with open(graph_file, 'wb') as f:
                pkl.dump([self.edge_index, self.edge_weight], f)

        else:
            with open(graph_file, 'rb') as f:
                self.edge_index, self.edge_weight = pkl.load(f)

    def get_train_test_portion(self, current_fold, mode):
        portion = []
        if mode == 'test':
            portion = self.query_ids_cv[self.args.test * current_fold:self.args.test * (current_fold + 1)]
        elif mode == 'train':
            portion = self.query_ids_cv[:self.args.test * current_fold] + self.query_ids_cv[
                                                                          self.args.test * (current_fold + 1):]
        return portion

    def get_eval_data(self, current_fold, mode):
        y_true = []
        query_portion = self.get_train_test_portion(current_fold, mode)
        query_embeddings = torch.Tensor([self.queries[query_id] for query_id in query_portion])
        document_embeddings = self.resource_document_embedding

        for query_id in query_portion:
            query_y_true = []
            for resource_id in self.resource_ids:
                score = self.resource_query_similarity[(self.resource_query_similarity['query_id'] == query_id) & (self.resource_query_similarity['resource_id'] == resource_id)]['similarity_score'].values[0]
                score = max(score, 0)
                query_y_true.append(score)
            y_true.append(query_y_true)

        return query_embeddings, query_portion, document_embeddings, y_true

    def get_positive_negative_resources(self):
        self.query2resources = {}

        for query_id in tqdm(self.query_ids):
            pos_resources = sorted(self.resource_query_similarity[
                                       (self.resource_query_similarity['query_id'] == query_id) & (self.resource_query_similarity['similarity_score'] > 0.0)]['resource_id'].tolist())
            pos_scores = [self.resource_query_similarity[(self.resource_query_similarity['query_id'] == query_id) & (
                        self.resource_query_similarity['resource_id'] == pos_resource_id)]['similarity_score'].values[0]
                          for pos_resource_id in pos_resources]
            pos_prob = [score / sum(pos_scores) for score in pos_scores]

            neg_resources = sorted(self.resource_query_similarity[
                                       (self.resource_query_similarity['query_id'] == query_id) & (
                                                   self.resource_query_similarity['similarity_score'] <= 0.0)][
                                       'resource_id'].tolist())

            self.query2resources[query_id] = {
                'pos_resources': pos_resources,
                'pos_scores': pos_scores,
                'pos_prob': pos_prob,
                'neg_resources': neg_resources
            }

    def get_train_pairs(self, current_fold, sample_idx):

        sample_file = f'samples/{self.args.dataset}_bias{self.args.bias}_fold{current_fold}_idx{sample_idx}_doc{self.args.num_doc}_num{self.args.train_pair_count}.pkl'

        if not os.path.exists(sample_file):

            train_pairs = []
            current_fold_queries = self.get_train_test_portion(current_fold, mode='train')

            for count in trange(self.args.train_pair_count):
                query_id = random.choice(current_fold_queries)

                pos_resources = self.query2resources[query_id]['pos_resources']

                if self.args.bias:
                    pos_prob = self.query2resources[query_id]['pos_prob']
                else:
                    pos_prob = [1 / len(pos_resources) for p in pos_resources]

                pos_resource = np.random.choice(pos_resources, p=pos_prob)

                neg_resources = self.query2resources[query_id]['neg_resources']
                neg_resource = random.choice(neg_resources)

                train_pairs.append((query_id, pos_resource, neg_resource))

            train_pairs = torch.tensor(train_pairs, dtype=torch.long)

            with open(sample_file, 'wb') as f:
                pkl.dump(train_pairs, f)

        else:
            with open(sample_file, 'rb') as f:
                train_pairs = pkl.load(f)

        return train_pairs