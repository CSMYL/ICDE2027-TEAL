import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import dgl
from dgl import DGLGraph
from collections import defaultdict
import random

from __init__ import data_root
from data.TabularDataset import TabularDataset
from data.utils import get_ds_info


class CSVToGraphAdapter(Dataset):
    """
    Converts TabularDataset into a Multiplex Graph structure as described in the TabGNN paper.

    Core implementation:
    1. Each sample is a node (Main_table type)
    2. Neighbors are found via connect keys
    3. Graph: center node(0) + neighbor nodes(1,2,...,N) + edges (various types)
    """

    def __init__(self, dataset_name, datapoint_ids=None, encoders=None,
                 connect_keys=None, max_neighbors_per_key=20):
        """
        Args:
            dataset_name: dataset name
            datapoint_ids: list of data point IDs (corresponding to CSV row indices)
            encoders: feature encoder configuration
            connect_keys: list of keys used to build multiplex graph edges.
                          e.g., ['user_id', 'category1'].
                          None: auto-detect all categorical features as connect keys.
            max_neighbors_per_key: max neighbors to find per connect key
        """
        self.dataset_name = dataset_name
        self.datapoint_ids = datapoint_ids
        self.connect_keys = connect_keys
        self.max_neighbors_per_key = max_neighbors_per_key

        self.tabular_dataset = TabularDataset(dataset_name=dataset_name,
                                             datapoint_ids=datapoint_ids,
                                             encoders=encoders)

        if self.tabular_dataset.encoders is not None:
            self.tabular_dataset.fit_feat_encoders()
            self.tabular_dataset.encode(self.tabular_dataset.feature_encoders)

        self.n_samples = len(self.tabular_dataset)

        self.raw_data = self.tabular_dataset.raw_data

        if datapoint_ids is None:
            if hasattr(self.tabular_dataset, 'datapoint_ids') and self.tabular_dataset.datapoint_ids is not None:
                self.datapoint_ids = list(self.tabular_dataset.datapoint_ids)
            else:
                self.datapoint_ids = list(self.raw_data.index)
        else:
            self.datapoint_ids = list(datapoint_ids)

        if self.connect_keys is None:
            self.connect_keys = self._auto_detect_connect_keys()

        self._build_graph_structure()

        self.db_info = self._create_db_info()

    def _auto_detect_connect_keys(self):
        """Auto-detect connect keys: use categorical features as connect keys."""
        connect_keys = []
        for col_info in self.tabular_dataset.columns:
            if col_info.get('type') == 'CATEGORICAL':
                col_name = col_info['name']
                if col_name in self.raw_data.columns and col_name != 'TARGET':
                    connect_keys.append(col_name)
        if len(connect_keys) == 0:
            print("  Warning: no categorical features found for connect keys, using similarity-based connections")
            connect_keys = ['__similarity__']
        return connect_keys

    def _build_graph_structure(self):
        """Build multiplex graph structure between samples."""
        self.connect_key_dict_list = []

        for key_idx, key in enumerate(self.connect_keys):
            if key == '__similarity__':
                self._build_similarity_connections(key_idx)
            else:
                key_dict = defaultdict(list)
                for idx, sample_id in enumerate(self.datapoint_ids):
                    try:
                        data_idx = idx + 1
                        if data_idx < len(self.raw_data):
                            key_value = self.raw_data.iloc[data_idx][key]
                        elif sample_id in self.raw_data.index:
                            key_value = self.raw_data.loc[sample_id, key]
                        else:
                            continue
                    except (KeyError, IndexError):
                        continue

                    if key_value == key:
                        continue

                    if pd.isna(key_value):
                        key_value = '__NA__'
                    key_dict[key_value].append(idx)
                self.connect_key_dict_list.append(key_dict)

    def _build_similarity_connections(self, key_idx):
        """Build connections based on feature similarity (for datasets without categorical features)."""
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        all_features = []
        for idx in range(len(self.tabular_dataset)):
            (cat_feats, cont_feats), _ = self.tabular_dataset[idx]

            if cat_feats is not None and len(cat_feats) > 0:
                if isinstance(cat_feats, torch.Tensor):
                    if isinstance(cont_feats, torch.Tensor):
                        feat_vec = torch.cat([cat_feats.flatten(), cont_feats.flatten()])
                    else:
                        feat_vec = cat_feats.flatten()
                else:
                    feat_vec = torch.cat([torch.tensor(cat_feats).flatten(), cont_feats.flatten()])
            else:
                if isinstance(cont_feats, torch.Tensor):
                    feat_vec = cont_feats.flatten()
                else:
                    feat_vec = torch.tensor(cont_feats).flatten()

            if isinstance(feat_vec, torch.Tensor):
                all_features.append(feat_vec.numpy())
            else:
                all_features.append(feat_vec)

        all_features = np.array(all_features)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(all_features)

        n_neighbors = min(self.max_neighbors_per_key + 1, len(all_features))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        knn.fit(features_scaled)
        distances, indices = knn.kneighbors(features_scaled)

        key_dict = defaultdict(list)
        for i, neighbors in enumerate(indices):
            key_dict['__similarity__'].extend([(i, j) for j in neighbors[1:]])

        self.connect_key_dict_list.append(key_dict)

    def _create_db_info(self):
        """Create db_info structure similar to DatabaseDataset."""
        node_types_and_features = {
            'Main_table': {}
        }

        node_types_and_features['Main_table']['INDEX_ID'] = {'type': 'NUMERIC'}

        for col_info in self.tabular_dataset.columns:
            col_name = col_info['name']
            col_type = col_info.get('type', 'NUMERIC')

            if col_type == 'CATEGORICAL':
                db_type = 'CATEGORICAL'
            elif col_type in ['NUMERIC', 'SCALAR']:
                db_type = 'SCALAR'
            else:
                db_type = 'SCALAR'

            node_types_and_features['Main_table'][col_name] = {'type': db_type}

        return {
            'node_types_and_features': node_types_and_features,
            'node_type_to_int': {'Main_table': 0},
            'edge_type_to_int': {'self': 0},
            'task': {
                'n_classes': 2 if self.tabular_dataset.ds_info['processed']['task'] == 'binary classification' else 1
            },
            'label_feature': 'Main_table.TARGET'
        }

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Returns a data point in a format fully consistent with DatabaseDataset.
        Returns:
            (edge_list, node_types, edge_types, features, label)

        Format:
        - edge_list: [(neighbor_idx, center_idx), ...]  # neighbors connect to center
        - node_types: [0, 0, 0, ...]  # all nodes are Main_table type (0)
        - edge_types: [1, 2, ...]  # different connect keys -> different edge types
        - features: {'Main_table': {'INDEX_ID': [...], 'feature_name': [...], ...}}
        - label: label value
        """
        sample_id = self.datapoint_ids[idx]
        input_data, label = self.tabular_dataset[idx]
        if isinstance(input_data, tuple):
            cat_feats, cont_feats = input_data
        else:
            cat_feats, cont_feats = None, None

        features = {
            'Main_table': {}
        }
        for col_info in self.tabular_dataset.columns:
            col_name = col_info['name']
            if col_name != 'TARGET':
                features['Main_table'][col_name] = []

        related_instance_list = []

        for key_idx, key in enumerate(self.connect_keys):
            if key == '__similarity__':
                related_samples = []
                if key_idx < len(self.connect_key_dict_list):
                    similarity_edges = self.connect_key_dict_list[key_idx].get('__similarity__', [])
                    for source, target in similarity_edges:
                        if source == idx:
                            related_samples.append(target)
                related_instance_list.append(related_samples[:self.max_neighbors_per_key])
            else:
                related_samples = []
                if key_idx < len(self.connect_key_dict_list):
                    key_dict = self.connect_key_dict_list[key_idx]
                    try:
                        data_idx = idx + 1
                        if data_idx < len(self.raw_data):
                            key_value = self.raw_data.iloc[data_idx][key]
                            if key_value == key:
                                key_value = None
                        elif sample_id in self.raw_data.index:
                            key_value = self.raw_data.loc[sample_id, key]
                        else:
                            key_value = None
                    except (KeyError, IndexError):
                        key_value = None

                    if key_value is not None:
                        if pd.isna(key_value):
                            key_value = '__NA__'
                        candidates = key_dict.get(key_value, [])
                        related_samples = [i for i in candidates if i != idx][:self.max_neighbors_per_key]
                related_instance_list.append(related_samples)

        all_node_indices = [idx]
        for related_list in related_instance_list:
            all_node_indices.extend(related_list)
        all_node_indices = list(set(all_node_indices))
        if idx in all_node_indices:
            all_node_indices.remove(idx)
        all_node_indices = [idx] + all_node_indices

        if len(all_node_indices) == 0:
            all_node_indices = [idx]

        node_id_to_graph_idx = {node_idx: graph_idx for graph_idx, node_idx in enumerate(all_node_indices)}

        edge_list = []
        edge_types = []

        center_graph_idx = 0

        for key_idx, related_list in enumerate(related_instance_list):
            edge_type = key_idx + 1
            for neighbor_idx in related_list:
                neighbor_graph_idx = node_id_to_graph_idx[neighbor_idx]
                edge_list.append((neighbor_graph_idx, center_graph_idx))
                edge_types.append(edge_type)

        node_types = [0] * len(all_node_indices)

        for graph_idx, node_idx in enumerate(all_node_indices):
            node_sample_id = self.tabular_dataset.datapoint_ids[node_idx]

            if 'INDEX_ID' not in features['Main_table']:
                features['Main_table']['INDEX_ID'] = []
            features['Main_table']['INDEX_ID'].append(int(node_sample_id))

            if node_sample_id in self.raw_data.index:
                for col_info in self.tabular_dataset.columns:
                    col_name = col_info['name']
                    if col_name != 'TARGET':
                        if col_name not in features['Main_table']:
                            features['Main_table'][col_name] = []
                        value = self.raw_data.loc[node_sample_id, col_name]
                        if pd.isna(value):
                            value = None
                        features['Main_table'][col_name].append(value)

        dp_id = sample_id
        return dp_id, (edge_list, node_types, edge_types, features, label)

    @property
    def feature_encoders(self):
        """Return feature encoders (for GNN model initialization)."""
        from data.data_encoders import NullEnc

        feature_encoders = {
            'Main_table': {}
        }

        feature_encoders['Main_table']['INDEX_ID'] = NullEnc()

        if hasattr(self.tabular_dataset, 'feature_encoders') and self.tabular_dataset.feature_encoders:
            for col_name in self.raw_data.columns:
                if col_name != 'TARGET':
                    if col_name in self.tabular_dataset.feature_encoders:
                        feature_encoders['Main_table'][col_name] = self.tabular_dataset.feature_encoders[col_name]
                    else:
                        feature_encoders['Main_table'][col_name] = NullEnc()
        else:
            for col_name in self.raw_data.columns:
                if col_name != 'TARGET':
                    feature_encoders['Main_table'][col_name] = NullEnc()

        return feature_encoders
