import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import ndcg_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='gov2', type=str, help='dataset')
    parser.add_argument("--doc_numbers_per_resource", default=10, type=int, help='number of docs per resource')
    parser.add_argument("--embedding_repo", default='ms-marco-MiniLM-L-12-v2', type=int, help='hugginface model name')
    parser.add_argument("--train_pair_count", default=10000, type=int, help='number of train pairs')
    parser.add_argument("--train", default=-1, type=int, help='number of train queries')
    parser.add_argument("--test", default=-1, type=int, help='number of test queries')
    parser.add_argument("--folds", default=-1, type=int, help='number of folds for cross validation')
    parser.add_argument("--random_seed", default=10, type=int, help='random seed')
    parser.add_argument("--gpu", default='0', type=str, help='gpu number')

    parser.add_argument("--ndcg_k", default=[1, 5, 10, 20], type=list, help='nDCG results at k slice')
    parser.add_argument("--np_k", default=[1, 5, 10, 20], type=list, help='nDCG results at k slice')

    parser.add_argument("--eval_test", default=True, type=bool, help='Evaluate test data in each epoch')
    parser.add_argument("--eval_train", default=True, type=bool, help='Evaluate train data in each epoch')

    parser.add_argument("--batch_size", default=256, type=int, help='batch size')
    parser.add_argument("--epochs", default=50, type=int, help='number of epochs')
    parser.add_argument("--learning_rate", default=1e-4, type=float, help='learning rate')
    parser.add_argument("--weight_decay", default=1e-5, type=float, help='reg weight')
    parser.add_argument("--dim", default=768, type=int, help='embedding dimension')

    parser.add_argument("--box_type", default="geometric", type=str, help='box embedding type')
    parser.add_argument("--gamma", default=0.0, type=float, help='box-vector distance parameter')
    parser.add_argument("--delta", default=1.0, type=float, help='margin in hinge loss')
    parser.add_argument("--threshold", default=0.8, type=float, help='resource-resource similarity threshold')

    parser.add_argument("--use_gnn", default=1, type=int, help='Use GNN')
    parser.add_argument("--bias", default=1, type=int, help='Sampling bias')
    parser.add_argument("--loss_type", default='hinge', type=str, help='loss function')

    return parser.parse_args()


def vector_box_distance(vector, center, offset, gamma=0.3):
    if vector.dim() == 1:
        vector = vector.unsqueeze(0)
        center = center.unsqueeze(0)
        offset = offset.unsqueeze(0)

    lower_left = center - offset
    upper_right = center + offset
    dist_out = torch.sum((torch.relu(vector - upper_right) + torch.relu(lower_left - vector)) ** 2, 1)
    dist_in = torch.sum((center - torch.min(upper_right, torch.max(lower_left, vector))) ** 2, 1)
    dist = dist_out + gamma * dist_in
    return dist


def bpr_loss(query_point, pos_center, pos_offset, neg_center, neg_offset):
    pos_distance = vector_box_distance(query_point, pos_center, pos_offset)
    neg_distance = vector_box_distance(query_point, neg_center, neg_offset)
    loss = - torch.log(torch.sigmoid(neg_distance - pos_distance))
    return loss.mean()


def hinge_loss(query_point, pos_center, pos_offset, neg_center, neg_offset, delta):
    pos_distance = vector_box_distance(query_point, pos_center, pos_offset)
    neg_distance = vector_box_distance(query_point, neg_center, neg_offset)
    loss = torch.relu(pos_distance - neg_distance + delta)
    return loss.mean()


def ndcg_eval(model, query_layer, data, current_fold, mode, args, device):
    ndcg_results = {}
    np_results = {}
    model.eval()

    with torch.no_grad():
        query_embeddings, document_embeddings, y_true = data.get_eval_data(current_fold, mode)
        query_embeddings = query_embeddings.to(device)
        document_embeddings = document_embeddings.to(device)

        center, offset = model(document_embeddings)

        # Compute distance between test query and data source boxes
        y_score = []
        for query_embedding in tqdm(query_embeddings):
            resource_y_score = []
            for resource_id in range(center.shape[0]):
                dist = vector_box_distance(query_layer(query_embedding), center[resource_id], offset[resource_id])
                resource_y_score.append(- dist.item())
            y_score.append(resource_y_score)

        for k in args.ndcg_k:
            ndcg_results[f"nDCG @{k}"] = ndcg_score(y_true, y_score, k=k)

        for k in args.np_k:
            np_results[f"nP @{k}"] = normalized_precision(y_true, y_score, k=k)

    return ndcg_results, np_results


def normalized_precision(y_true, y_score, k):
    nPatk = []

    def get_dict_data(list_data):
        dict_data = {}
        for qid, docs in enumerate(list_data):
            dict_data[f"q{qid}"] = {}
            for rid, doc in enumerate(docs):
                dict_data[f"q{qid}"][f"r{rid}"] = float(doc)
        return dict_data

    idealscores = get_dict_data(y_true)
    predictedscores = get_dict_data(y_score)

    for qID in idealscores:
        for SEID in predictedscores[qID]:
            if not SEID in idealscores[qID]:
                idealscores[qID][SEID] = 0.
    for qID in sorted(idealscores.keys()):
        tmp = sorted(predictedscores[qID].items(), key=lambda x: (x[1], x[0]), reverse=True)
        sorted_predictedSEIDs = [t[0] for t in tmp]
        top_predictedSEIDs = sorted_predictedSEIDs[:k]
        tmp = sorted(idealscores[qID].items(), key=lambda x: (x[1], x[0]), reverse=True)
        sorted_idealSEIDs = [t[0] for t in tmp]
        top_idealSEIDs = sorted_idealSEIDs[:k]
        top_predicted_values = [idealscores[qID][SEID] for SEID in top_predictedSEIDs]
        top_ideal_values = [idealscores[qID][SEID] for SEID in top_idealSEIDs]
        nPatk.append(np.sum(top_predicted_values) / np.sum(top_ideal_values))

    return sum(nPatk) / len(nPatk)