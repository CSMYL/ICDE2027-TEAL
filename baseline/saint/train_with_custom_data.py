import torch
from torch import nn
from models import SAINT
from data_adapter import create_saint_datasets_from_reconstruct, load_dataset_by_name, get_dataset_info
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, classification_scores, mean_sq_error, classification_scores_with_timing, mean_sq_error_with_timing
from augmentations import embed_data_mask
import os
import numpy as np
import time

parser = argparse.ArgumentParser()

# Data related arguments
parser.add_argument('--dataset', default='housesale', type=str, choices=['adult', 'cardio', 'creditcard', 'diamonds', 'elevator', 'housesale', 'crime', 'meps', 'mslr'],
                   help='Dataset name to use')
parser.add_argument('--data_path', type=str, default=None, help='Optional custom path to CSV file')
parser.add_argument('--target_col_idx', type=int, default=None, help='Target column index (default: last column)')
parser.add_argument('--task', type=str, default=None, choices=['binary', 'multiclass', 'regression'],
                   help='Task type (auto-determined if not specified)')
parser.add_argument('--max_samples', type=int, default=100000, help='Subsample size (only used for MSLR-WEB10K; 0 = full data)')

# Model arguments
parser.add_argument('--embedding_size', default=64, type=int)
parser.add_argument('--transformer_depth', default=4, type=int)
parser.add_argument('--attention_heads', default=4, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument('--attentiontype', default='colrow', type=str, choices=['col', 'colrow', 'row', 'justmlp', 'attn', 'attnmlp'])

# Training arguments
parser.add_argument('--optimizer', default='AdamW', type=str, choices=['AdamW', 'Adam', 'SGD'])
parser.add_argument('--lr', default=0.00001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
parser.add_argument('--run_name', default='testrun', type=str)
parser.add_argument('--set_seed', default=42, type=int)

opt = parser.parse_args()

dataset_info = get_dataset_info(opt.dataset)
if opt.task is None:
    opt.task = dataset_info['task']
if opt.target_col_idx is None:
    opt.target_col_idx = dataset_info['default_target']

modelsave_path = os.path.join(os.getcwd(), opt.savemodelroot, opt.task, opt.dataset, opt.run_name)
if opt.task == 'regression':
    opt.dtask = 'reg'
else:
    opt.dtask = 'clf'

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")

torch.manual_seed(opt.set_seed)
os.makedirs(modelsave_path, exist_ok=True)

print(f'Loading and processing the {opt.dataset} dataset...')
print(f'Dataset info: {dataset_info}')

# Load dataset
try:
    dataset, v, num_classes_dict = load_dataset_by_name(opt.dataset, opt.data_path, opt.max_samples)
    print(f"\u2713 Dataset loaded successfully")
    print(f"  Total features: {len(v)}")
    print(f"  Categorical features: {sum(v)}")
    print(f"  Continuous features: {len(v) - sum(v)}")
except Exception as e:
    print(f"\u2717 Failed to load dataset: {e}")
    exit(1)

# Create SAINT format datasets
try:
    train_ds, valid_ds, test_ds, cat_dims, cat_idxs, con_idxs, continuous_mean_std = create_saint_datasets_from_reconstruct(
        dataset, v, num_classes_dict, opt.target_col_idx, opt.task
    )
    print(f"\u2713 SAINT datasets created successfully")
except Exception as e:
    print(f"\u2717 Failed to create SAINT datasets: {e}")
    exit(1)

# Create data loaders
trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, num_workers=0)
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False, num_workers=0)
testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False, num_workers=0)

# Set target dimension
y_dim = 1 if opt.task == 'regression' else (2 if opt.task == 'binary' else len(np.unique(train_ds.y)))
# Add one dimension for CLS token
cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int)

print(f"\nDataset details:")
print(f"  Categorical features: {len(cat_idxs)}")
print(f"  Continuous features: {len(con_idxs)}")
print(f"  Target dimension: {y_dim}")
print(f"  Training samples: {len(train_ds)}")
print(f"  Validation samples: {len(valid_ds)}")
print(f"  Test samples: {len(test_ds)}")
print(f"  Categorical dimensions: {cat_dims}")

# Create model
model = SAINT(
    categories=tuple(cat_dims),
    num_continuous=len(con_idxs),
    dim=opt.embedding_size,
    dim_out=1,
    depth=opt.transformer_depth,
    heads=opt.attention_heads,
    attn_dropout=opt.attention_dropout,
    ff_dropout=opt.ff_dropout,
    mlp_hidden_mults=(4, 2),
    cont_embeddings='MLP',
    attentiontype=opt.attentiontype,
    final_mlp_style='sep',
    y_dim=y_dim
)

# Set loss function
if opt.task == 'regression':
    criterion = nn.MSELoss().to(device)
elif opt.task == 'binary':
    criterion = nn.CrossEntropyLoss().to(device)
elif opt.task == 'multiclass':
    criterion = nn.CrossEntropyLoss().to(device)
else:
    if y_dim == 2:
        criterion = nn.CrossEntropyLoss().to(device)
    elif y_dim > 2:
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        raise Exception(f'Unsupported task type: {opt.task} with y_dim: {y_dim}')

model.to(device)

# Select optimizer
if opt.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    from utils import get_scheduler
    scheduler = get_scheduler(opt, optimizer)
elif opt.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
elif opt.optimizer == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)

best_valid_auroc = 0
best_valid_accuracy = 0
best_test_auroc = 0
best_test_accuracy = 0
best_valid_rmse = 100000

print('\nTraining begins now...')
for epoch in range(opt.epochs):
    epoch_start_time = time.time()
    train_start_time = time.time()
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
        _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, False)
        reps = model.transformer(x_categ_enc, x_cont_enc)
        y_reps = reps[:, 0, :]
        y_outs = model.mlpfory(y_reps)
        if opt.task == 'regression':
            loss = criterion(y_outs, y_gts)
        else:
            loss = criterion(y_outs, y_gts.squeeze())
        loss.backward()
        optimizer.step()
        if opt.optimizer == 'SGD':
            scheduler.step()
        running_loss += loss.item()
    train_time = time.time() - train_start_time
    if epoch % 1 == 0:
        valid_start_time = time.time()
        model.eval()
        with torch.no_grad():
            if opt.task in ['binary', 'multiclass']:
                accuracy, auroc, valid_avg_forward = classification_scores_with_timing(model, validloader, device, opt.task, False)
                test_accuracy, test_auroc, test_avg_forward = classification_scores_with_timing(model, testloader, device, opt.task, False)
                print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f, AVG FORWARD TIME: %.4fs' % (epoch + 1, accuracy, auroc, valid_avg_forward))
                print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.4f, AVG FORWARD TIME: %.4fs' % (epoch + 1, test_accuracy, test_auroc, test_avg_forward))
                if opt.task == 'multiclass':
                    if accuracy > best_valid_accuracy:
                        best_valid_accuracy = accuracy
                        best_test_auroc = test_auroc
                        best_test_accuracy = test_accuracy
                        torch.save(model.state_dict(), '%s/bestmodel.pth' % modelsave_path)
                else:
                    if auroc > best_valid_auroc:
                        best_valid_auroc = auroc
                        best_test_auroc = test_auroc
                        best_test_accuracy = test_accuracy
                        torch.save(model.state_dict(), '%s/bestmodel.pth' % modelsave_path)
            else:
                valid_rmse, valid_avg_forward = mean_sq_error_with_timing(model, validloader, device, False)
                test_rmse, test_avg_forward = mean_sq_error_with_timing(model, testloader, device, False)
                print('[EPOCH %d] VALID RMSE: %.4f, AVG FORWARD TIME: %.4fs' % (epoch + 1, valid_rmse, valid_avg_forward))
                print('[EPOCH %d] TEST RMSE: %.4f, AVG FORWARD TIME: %.4fs' % (epoch + 1, test_rmse, test_avg_forward))
                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    best_test_rmse = test_rmse
                    torch.save(model.state_dict(), '%s/bestmodel.pth' % modelsave_path)
        valid_time = time.time() - valid_start_time
        model.train()
    epoch_time = time.time() - epoch_start_time
    print(f'[EPOCH {epoch + 1}] TRAIN TIME: {train_time:.2f}s, VALID TIME: {valid_time:.2f}s, TOTAL TIME: {epoch_time:.2f}s')

total_parameters = count_parameters(model)
print('\n' + '='*50)
print('TRAINING COMPLETED')
print('='*50)
print('TOTAL NUMBER OF PARAMS: %d' % total_parameters)
if opt.task == 'binary':
    print('AUROC on best model: %.3f' % best_test_auroc)
elif opt.task == 'multiclass':
    print('Accuracy on best model: %.3f' % best_test_accuracy)
else:
    print('RMSE on best model: %.3f' % best_test_rmse)
print(f'Model saved to: {modelsave_path}/bestmodel.pth') 