import torch
from torch.utils.data import DataLoader, TensorDataset
from box_embedding import Model, QueryModel
import torch.optim as optim
from tqdm import tqdm
import random
import warnings
import numpy as np
import utils
from data import Dataset
from dataset_configs import config

warnings.filterwarnings('ignore')

args = utils.parse_args()

d_config = config[args.dataset]
args.train = d_config['train']
args.test = d_config['test']
args.folds = d_config['folds']

########## Fix Seeds ##########
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

########## GPU Settings ##########
if torch.cuda.is_available():
    device = torch.device("cuda:" + args.gpu)
else:
    device = torch.device("cpu")
print('Device:\t', device, '\n')

config = f'{args.dataset}_lr{args.learning_rate}_wd{args.weight_decay}_dim{args.dim}_gamma{args.gamma}_bias{args.bias}_loss-{args.loss_type}'

if args.loss_type == 'hinge':
    config += f'_delta{args.delta}'

if args.use_gnn:
    config += f'_gnn{args.threshold}'

config += f'_ver{args.version}'

print(config, '\n')


def train(model, query_layer, data, current_fold):
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)

    train_losses, train_results, test_results = [], {'ndcg': [], 'np': []}, {'ndcg': [], 'np': []}
    sample_cnt = 0

    model.train()
    for epoch in range(args.epochs):

        if epoch % 5 == 0:
            sample_cnt += 1
            train_pairs = data.get_train_pairs(current_fold, sample_cnt)
            train_data = TensorDataset(train_pairs)
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()

            # Get resource embeddings
            resource_center, resource_offset = model(data.resource_document_embedding.to(device))

            batch_data = batch[0]
            query_idx_batch, pos_idx_batch, neg_idx_batch = batch_data[:, 0], batch_data[:, 1], batch_data[:, 2]

            query_embedding_batch = torch.Tensor([list(data.queries[query_id]) for query_id in query_idx_batch])
            pos_center, pos_offset = resource_center[pos_idx_batch], resource_offset[pos_idx_batch]
            neg_center, neg_offset = resource_center[neg_idx_batch], resource_offset[neg_idx_batch]

            query_embedding_batch = query_layer(query_embedding_batch.to(device))

            if args.loss_type == 'hinge':
                loss = utils.hinge_loss(query_embedding_batch, pos_center, pos_offset, neg_center, neg_offset,
                                        args.delta)
            elif args.loss_type == 'bpr':
                loss = utils.bpr_loss(query_embedding_batch, pos_center, pos_offset, neg_center, neg_offset)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Average Loss: {epoch_loss / len(train_pairs)}")
        train_losses.append(epoch_loss / len(train_pairs))

        if args.eval_train:
            ndcg_results, np_results = utils.ndcg_eval(model, query_layer, data, current_fold, mode='train', args=args,
                                                       device=device, config=config)

            for k in args.ndcg_k:
                print(f'Train nDCG@{k}: {ndcg_results[f"nDCG @{k}"]:.6f}')
            train_results['ndcg'].append([ndcg_results[f"nDCG @{k}"] for k in args.ndcg_k])

            for k in args.np_k:
                print(f'Train nP@{k}: {np_results[f"nP @{k}"]:.6f}')
            train_results['np'].append([np_results[f"nP @{k}"] for k in args.np_k])

        if args.eval_test:
            if epoch == args.epochs - 1 and args.save_model:
                write_result = True
            else:
                write_result = False

            ndcg_results, np_results = utils.ndcg_eval(model, query_layer, data, current_fold, mode='test', args=args,
                                                       device=device, config=config, write_result=write_result)

            for k in args.ndcg_k:
                print(f'Test nDCG@{k}: {ndcg_results[f"nDCG @{k}"]:.6f}')
            test_results['ndcg'].append([ndcg_results[f"nDCG @{k}"] for k in args.ndcg_k])

            for k in args.np_k:
                print(f'Test nP@{k}: {np_results[f"nP @{k}"]:.6f}')
            test_results['np'].append([np_results[f"nP @{k}"] for k in args.np_k])

        print()

    with open(f'logs/{config}_fold{current_fold}.txt', 'w') as f:
        for epoch in range(args.epochs):
            log = ' '.join([f'{train_losses[epoch]}'] +
                           [str(x) for x in train_results['ndcg'][epoch] + test_results['ndcg'][epoch]] +
                           [str(x) for x in train_results['np'][epoch] + test_results['np'][epoch]])
            f.write(log + '\n')


def main():
    data = Dataset(args, device)

    # Train n number of folds for cross validation
    for current_fold in range(args.folds):
        print(f"************************ FOLD {current_fold + 1} *******************************")
        model = Model(args.box_type, args.dim, data.edge_index.to(device), data.edge_weight.float().to(device),
                      args.use_gnn).to(device)
        query_layer = QueryModel(args.dim).to(device)
        train(model, query_layer, data, current_fold)


if __name__ == "__main__":
    main()
