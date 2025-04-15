from torch_geometric.data import Data
import torch
from torch_geometric.io import fs,read_txt_array
from typing import Dict, Tuple
import os.path as osp
import numpy as np
# class HyperData(Data):
#     def __init__(self, x, edge_index, y=None):
#         super(HyperData, self).__init__()
#         self.x = x
#         self.edge_index = edge_index
#         self.y = y
#         self.num_nodes = x.size(0)
#         self.num_edges = max(edge_index[1]) + 1
#         self.num_node_features = x.size(1)
#         self.train_mask=None
#         self.val_mask=None
#         self.test_mask=None
import torch
import numpy as np
from torch_geometric.data import Data


def read_our_data(
    root: str,
    data_type='graph',
    folder='MUTAG',
    name='MUTAG',
    degree_as_tag=True,
):
    x_list = []
    with open(f"{root}/{data_type}/{folder}/{name}.txt", "r") as f:
        n_g = int(f.readline().strip())
        for _ in range(n_g):
            row = f.readline().strip().split()
            num_v, num_e = int(row[0]), int(row[1])
            g_lbl = [int(x) for x in row[2:]]
            v_lbl = f.readline().strip().split()
            v_lbl = [[int(x) for x in s.split("/")] for s in v_lbl]
            v_e_index = []
            e_cnt = 0
            for _ in range(num_e):
                row = f.readline().strip().split()
                if data_type == 'graph':
                    src, dst = int(row[0]), int(row[1])
                    if src>dst:
                        v_e_index.extend([[src, e_cnt], [dst, e_cnt]])
                        e_cnt+=1
                elif data_type=='hypergraph':
                    for d in row:
                        v_e_index.append([int(d),e_cnt])
                        e_cnt+=1
                else:
                    raise ImportError
            edge_index = torch.tensor(v_e_index, dtype=torch.long).t()
            # v_ft=[]
            # data = Data(
            #     x=v_ft,
            #     edge_index=x["edge_index"],
            #     edge_attr=torch.tensor(x["e_lbl"], dtype=torch.long),
            #     y=torch.tensor(x["g_lbl"], dtype=torch.long)
            # )      
            def calculate_node_degree(edge_index):
                    # 获取节点的最大索引，加 1 得到节点的总数
                num_nodes = edge_index[0].max().item() + 1
                # 初始化一个全零的张量来存储每个节点的度
                degrees = torch.zeros(num_nodes, dtype=torch.long)
                # 统计每个节点作为起点和终点出现的次数
                degrees.index_add_(0, edge_index[0], torch.ones_like(edge_index[0]))
                #degrees.index_add_(0, edge_index[1], torch.ones_like(edge_index[1]))
                return degrees
            deg=calculate_node_degree(edge_index)      
            x_list.append(
                {
                    'num_v': num_v,
                    'num_e': num_e,
                    'g_lbl': g_lbl,
                    'v_lbl': v_lbl,
                    'deg': deg,
                    'edge_index': edge_index
                }
            )

    # 示例 edge_index
    
    for x in x_list:
        if degree_as_tag:
            x["v_lbl"] = [int(v) for v in x["deg"]]
        # if isinstance(x["dhg"], Graph):
        #     x["e_lbl"] = [2] * x["num_e"]
        # else:
        #     x["e_lbl"] = [int(e) for e in x["dhg"].deg_e]    
    for x in x_list:
        x['e_lbl'] = [2] * x['num_e']
    v_lbl_set, e_lbl_set, g_lbl_set = set(), set(), set()
    for x in x_list:
        if isinstance(x["v_lbl"][0], list):
            for v_lbl in x["v_lbl"]:
                v_lbl_set.update(v_lbl)
        else:
            v_lbl_set.update(x["v_lbl"])
        e_lbl_set.update(x["e_lbl"])
        g_lbl_set.update(x["g_lbl"])
    # re-map labels
    v_lbl_map = {x: i for i, x in enumerate(sorted(v_lbl_set))}
    e_lbl_map = {x: i for i, x in enumerate(sorted(e_lbl_set))}
    g_lbl_map = {x: i for i, x in enumerate(sorted(g_lbl_set))}
    ft_dim, n_classes = len(v_lbl_set), len(g_lbl_set)
    import pdb
    #pdb.set_trace()
    data_list = []
    for x in x_list:
        x["g_lbl"] = [g_lbl_map[c] for c in x["g_lbl"]]
        if isinstance(x["v_lbl"][0], list):
            x["v_lbl"] = [tuple(sorted([v_lbl_map[c] for c in s])) for s in x["v_lbl"]]
        else:
            x["v_lbl"] = [v_lbl_map[c] for c in x["v_lbl"]]
        x["e_lbl"] = [e_lbl_map[c] for c in x["e_lbl"]]
        # 进行独热编码
        v_ft = torch.zeros((x["num_v"], ft_dim), dtype=torch.float)
        for v_idx, v_lbls in enumerate(x["v_lbl"]):
            if isinstance(v_lbls, (list, tuple)):
                for v_lbl in v_lbls:
                    v_ft[v_idx, v_lbl] = 1
            else:
                v_ft[v_idx, v_lbls] = 1
        # import pdb
        # pdb.set_trace()
        # 创建 Data 对象
        data = Data(
            x=v_ft,
            edge_index=x["edge_index"],
            #edge_attr=torch.tensor(x["e_lbl"], dtype=torch.long),
            y=torch.tensor(x["g_lbl"], dtype=torch.long)
        )
        data_list.append(data)
    return data_list

import os
from torch_geometric.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, root=None, data_type='graph', folder='MUTAG', name='MUTAG', transform=None, pre_transform=None):
        if root==None:
            self._root = os.getcwd()
        else:
            self._root = root
        self.data_type = data_type
        self.folder = folder
        self.name = name
        
        super().__init__(self._root, transform, pre_transform)
        self.data_list=torch.load(os.path.join(self.processed_dir, 'data.pt'))
    @property
    def raw_file_names(self):
        return [f"{self.name}.txt"]
    @property
    def processed_dir(self):
        return os.path.join(self._root, self.data_type, self.folder, self.name)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = read_our_data(self._root, self.data_type, self.folder, self.name)
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        torch.save(data_list, os.path.join(self.processed_dir, 'data.pt'))
    def len(self):
        return len(torch.load(os.path.join(self.processed_dir, 'data.pt')))

    def get(self, idx):
        #data_list = torch.load(os.path.join(self.processed_dir, 'data.pt'))
        return self.data_list[idx]

    def __repr__(self):
        data_list = torch.load(os.path.join(self.processed_dir, 'data.pt'))
        num_graphs = len(data_list)
        node_feature_dim = data_list[0].x.size(1) if data_list else 0
        class_set = set()
        for data in data_list:
            class_set.update(data.y.tolist())
        num_classes = len(class_set)
        return f'CustomDataset(Num graphs: {num_graphs}, Node feature dimensions: {node_feature_dim}, Classes: {num_classes})'

    