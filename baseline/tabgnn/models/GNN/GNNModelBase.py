import time
import pdb

import numpy as np
import torch
from dgl import DGLGraph
# In DGL 0.9+, BatchedDGLGraph has been removed; dgl.batch() returns DGLGraph.
# For compatibility, use DGLGraph as the type alias.
try:
    from dgl import BatchedDGLGraph
except ImportError:
    BatchedDGLGraph = DGLGraph
from torch import nn

import models.tabular as tab_models
from data.utils import get_db_info
from models import losses, activations
from models import readouts


class SafeBatchNorm1d(nn.Module):
    """
    Safe BatchNorm1d wrapper that falls back to LayerNorm1d when batch_size=1.
    This avoids BatchNorm errors with batch_size=1.
    """
    def __init__(self, batchnorm):
        super().__init__()
        self.batchnorm = batchnorm
        self.layernorm = nn.LayerNorm(batchnorm.num_features)
        self.use_layernorm = False

    def forward(self, x):
        if x.size(0) == 1:
            if not self.use_layernorm:
                self.layernorm.weight.data = self.batchnorm.weight.data.clone()
                self.layernorm.bias.data = self.batchnorm.bias.data.clone()
                self.use_layernorm = True
            return self.layernorm(x)
        else:
            self.use_layernorm = False
            return self.batchnorm(x)


class GNNModelBase(nn.Module):
    """Base class for all GNN models."""

    def __init__(self, writer, dataset_name, feature_encoders, hidden_dim, init_model_class_name, init_model_kwargs,
                 n_layers, activation_class_name, activation_class_kwargs, norm_class_name, norm_class_kwargs,
                 loss_class_kwargs, loss_class_name, p_dropout, readout_class_name, readout_kwargs, fcout_layer_sizes,
                 use_jknet, cat_fz_embedding, use_readout=True):
        super(GNNModelBase, self).__init__()
        self.writer = writer
        self.db_info = get_db_info(dataset_name)
        self.n_out = self.db_info['task']['n_classes']
        self.feature_encoders = feature_encoders
        self.init_model_class = tab_models.__dict__[init_model_class_name]
        self.init_model_kwargs = init_model_kwargs
        self.hidden_dim = hidden_dim
        self.p_dropout = p_dropout
        self.n_layers = n_layers
        if loss_class_kwargs.get('weight', None):
            loss_class_kwargs['weight'] = torch.Tensor(loss_class_kwargs['weight'])
        self.act_class = activations.__dict__[activation_class_name]
        self.act_class_kwargs = activation_class_kwargs
        self.norm_class = nn.__dict__[norm_class_name]
        self.norm_class_kwargs = norm_class_kwargs
        self.loss_fxn = losses.__dict__[loss_class_name](self, **loss_class_kwargs)
        self.use_jknet = use_jknet
        self.cat_fz_embedding = cat_fz_embedding
        self.use_readout = use_readout

        self.node_initializers = nn.ModuleDict()
        self.node_init_info = {}
        for node_type, features in self.db_info['node_types_and_features'].items():
            cat_feat_origin_cards = []
            cont_feat_origin = []
            for feature_name, feature_info in features.items():
                if '{}.{}'.format(node_type, feature_name) != self.db_info['label_feature']:
                    enc = self.feature_encoders[node_type][feature_name]
                    cat_feat_origin_cards += [(f'{feature_name}_{i}', card) for i, card in enumerate(enc.cat_cards)]
                    cont_feat_origin += [feature_name] * enc.cont_dim
            self.node_init_info[node_type] = {
                'cat_feat_origin_cards': cat_feat_origin_cards,
                'cont_feat_origin': cont_feat_origin,
            }
            self.node_initializers[node_type] = self.init_model_class(writer=writer,
                                                                      dataset_name=None,
                                                                      n_cont_features=len(cont_feat_origin),
                                                                      cat_feat_origin_cards=cat_feat_origin_cards,
                                                                      n_out=hidden_dim,
                                                                      **self.init_model_kwargs)

        if self.use_jknet:
            self.readout = readouts.__dict__[readout_class_name](hidden_dim=hidden_dim*n_layers, **readout_kwargs)
        else:
            self.readout = readouts.__dict__[readout_class_name](hidden_dim=hidden_dim, **readout_kwargs)

        if all(isinstance(s, float) for s in fcout_layer_sizes):
            fcout_layer_sizes = [int(self.hidden_dim * s) for s in fcout_layer_sizes]
        assert all(isinstance(s, int) for s in fcout_layer_sizes)
        self.layer_sizes = fcout_layer_sizes

        prev_layer_size = self.hidden_dim
        if self.use_jknet:
            prev_layer_size += self.hidden_dim*(n_layers-1)

        if self.cat_fz_embedding:
            prev_layer_size += self.hidden_dim

        fcout_layers = []
        for layer_size in self.layer_sizes:
            fcout_layers.append(nn.Linear(prev_layer_size, layer_size))
            fcout_layers.append(self.get_act())
            fcout_layers.append(self.get_norm(layer_size))
            fcout_layers.append(nn.Dropout(self.p_dropout))
            prev_layer_size = layer_size
        fcout_layers.append(nn.Linear(prev_layer_size, self.n_out))
        self.fcout = nn.Sequential(*fcout_layers)

    def get_act(self):
        return self.act_class(**self.act_class_kwargs)

    def get_norm(self, num_feats):
        """
        Get normalization layer, automatically handling BatchNorm with batch_size=1.
        If BatchNorm1d is used, wraps it in SafeBatchNorm1d for automatic fallback.
        """
        norm = self.norm_class(num_feats, **self.norm_class_kwargs)

        if isinstance(norm, nn.BatchNorm1d):
            return SafeBatchNorm1d(norm)

        return norm

    def init_batch(self, bdgl: BatchedDGLGraph, b_features):
        """
        Uses the tabular models in self.node_initializers to encode the raw database features
        (datetimes, text, etc.) of each table, such that all nodes in bdgl have the same hidden
        state size.

        This method is run before self.gnn_forward.
        """
        b_node_types = bdgl.ndata['node_types']
        bdgl.ndata['h'] = torch.empty(bdgl.number_of_nodes(), self.hidden_dim, device=b_node_types.device)
        bdgl.ndata['h'][:] = np.nan
        for node_type, collated_features in b_features.items():
            node_features = self.node_initializers[node_type](collated_features)

            node_type_int = self.db_info['node_type_to_int'][node_type]
            idxs_this_node_type = (b_node_types == node_type_int).nonzero()[:, 0]
            bdgl.nodes[idxs_this_node_type].data['h'] = node_features

        return bdgl

    def forward(self, input):
        """Returns logits for output classes."""
        bdgl, features, main_node_ids = input
        g = self.init_batch(bdgl, features)

        fz_embedding = None
        out = self.gnn_forward(g, main_node_ids)

        return out

    def gnn_forward(self, g: BatchedDGLGraph, features):
        """
        Runs the GNN component of the model and returns logits for output classes.

        :param g: BatchedDGLGraph with g.ndata[h] initialized to (n_nodes x hidden_dim) tensor
                  by self.init_batch
        """
        raise NotImplementedError

    def pred_from_output(self, output):
        """Returns the model's class prediction given the output of self.forward."""
        return output.max(dim=1, keepdim=True)[1]
