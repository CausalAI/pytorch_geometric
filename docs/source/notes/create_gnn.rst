图网络消息传递框架
=================================

将卷积算子推广到不规则域 is typically expressed as a *neighborhood aggregation* or *message passing* scheme.
我们用 :math:`\mathbf{x}^{(k-1)}_i \in \mathbb{R}^F` 表示第 :math:`(k-1)` 层 node :math:`i` 的节点特征， :math:`\mathbf{e}_{j,i} \in \mathbb{R}^D` 表示从node :math:`j` 到 node :math:`i` 的边特征， 则消息传递图神经网络可以描述为

.. math::
  \mathbf{x}_i^{(k)} = \gamma^{(k)} \left( \mathbf{x}_i^{(k-1)}, \square_{j \in \mathcal{N}(i)} \, \phi^{(k)}\left(\mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)},\mathbf{e}_{j,i}\right) \right),

其中 :math:`\square` 表示具有置换不变性的可微函数(例如 sum, mean or max, and :math:`\gamma`)， :math:`\phi` 表示可微函数(例如多层感知机 MLPs).

.. contents::
    :local:

"MessagePassing" 基类
-------------------------------

PyTorch Geometric 提供了 :class:`torch_geometric.nn.MessagePassing` 基类，来帮助创建消息传递图神经网络。因此我们只需要指定函数 :math:`\phi` , *i.e.* :meth:`message`, and :math:`\gamma` , *.i.e.* :meth:`update`, as well as the aggregation scheme to use, *.i.e.* :obj:`aggr='add'`, :obj:`aggr='mean'` or :obj:`aggr='max'`.

This is done with the help of the following methods:

* :obj:`torch_geometric.nn.MessagePassing(aggr="add", flow="source_to_target")`: Defines the aggregation scheme to use (:obj:`"add"`, :obj:`"mean"` or :obj:`"max"`) and the flow direction of message passing (either :obj:`"source_to_target"` or :obj:`"target_to_source"`).
* :obj:`torch_geometric.nn.MessagePassing.propagate(edge_index, size=None, dim=0, **kwargs)`:
  The initial call to start propagating messages.
  Takes in the edge indices and all additional data which is needed to construct messages and to update node embeddings.
  Note that :meth:`propagate` is not limited to exchange messages in symmetric adjacency matrices of shape :obj:`[N, N]` only, but can also exchange messages in general sparse assignment matrices, *.e.g.*, bipartite graphs, of shape :obj:`[N, M]` by passing :obj:`size=(N, M)` as an additional argument.
  If set to :obj:`None`, the assignment matrix is assumed to be symmetric.
  For bipartite graphs with two independent sets of nodes and indices, and each set holding its own information, this split can be marked by passing the information as a tuple, *e.g.* :obj:`x=(x_N, x_M)`.
  Furthermore, the :obj:`dim` attribute indicates along which axis to propagate.
* :meth:`torch_geometric.nn.MessagePassing.message`: Constructs messages to node :math:`i` in analogy to :math:`\phi` for each edge in :math:`(j,i) \in \mathcal{E}` if :obj:`flow="source_to_target"` and :math:`(i,j) \in \mathcal{E}` if :obj:`flow="target_to_source"`.
  Can take any argument which was initially passed to :meth:`propagate`.
  In addition, tensors passed to :meth:`propagate` can be mapped to the respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
* :meth:`torch_geometric.nn.MessagePassing.update`: Updates node embeddings in analogy to :math:`\gamma` for each node :math:`i \in \mathcal{V}`.
  Takes in the output of aggregation as first argument and any argument which was initially passed to :meth:`propagate`.

Let us verify this by re-implementing two popular GNN variants, the `GCN layer from Kipf and Welling <https://arxiv.org/abs/1609.02907>`_ and the `EdgeConv layer from Wang et al. <https://arxiv.org/abs/1801.07829>`_.

GCN层实现
--------------------------


`GCN layer <https://arxiv.org/abs/1609.02907>`_ 在数学上定义为

.. math::

    \mathbf{x}_i^{(k)} = \sum_{j \in \mathcal{N}(i) \cup \{ i \}} \frac{1}{\sqrt{\deg(i)} \cdot \sqrt{deg(j)}} \cdot \left( \mathbf{\Theta} \cdot \mathbf{x}_j^{(k-1)} \right),

where neighboring node features are first transformed by a weight matrix :math:`\mathbf{\Theta}`, normalized by their degree, and finally summed up.
该公式可以分为以下步骤：

1. Add self-loops to the adjacency matrix.
2. Linearly transform node feature matrix.
3. Compute normalization coefficients.
4. Normalize node features in :math:`\phi`.
5. Sum up neighboring node features (:obj:`"add"` aggregation).
6. Return new node embeddings in :math:`\gamma`.


通常在消息传递发生之前计算步骤1-3。使用 :class:`torch_geometric.nn.MessagePassing` 基类可以轻松地执行步骤4-6。完整的层实现如下所示：


.. code-block:: python

    import torch
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops, degree

    class GCNConv(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
            self.lin = torch.nn.Linear(in_channels, out_channels)

        def forward(self, x, edge_index):
            # x has shape [N, in_channels]
            # edge_index has shape [2, E]

            # Step 1: Add self-loops to the adjacency matrix.
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

            # Step 2: Linearly transform node feature matrix.
            x = self.lin(x)

            # Step 3: Compute normalization
            row, col = edge_index
            deg = degree(row, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            # Step 4-6: Start propagating messages.
            return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                                  norm=norm)

        def message(self, x_j, norm):
            # x_j has shape [E, out_channels]

            # Step 4: Normalize node features.
            return norm.view(-1, 1) * x_j

        def update(self, aggr_out):
            # aggr_out has shape [N, out_channels]

            # Step 6: Return new node embeddings.
            return aggr_out

:class:`GCNConv` inherits from :class:`torch_geometric.nn.MessagePassing` with :obj:`"add"` propagation.
All the logic of the layer takes place in :meth:`forward`.
Here, we first add self-loops to our edge indices using the :meth:`torch_geometric.utils.add_self_loops` function (step 1), as well as linearly transform node features by calling the :class:`torch.nn.Linear` instance (step 2).

We then proceed to call :meth:`propagate`, which internally calls the :meth:`message` and :meth:`update` functions.
As additional arguments for message propagation, we pass the node embeddings :obj:`x`.

In the :meth:`message` function, we need to normalize the neighboring node features :obj:`x_j`.
Here, :obj:`x_j` denotes a *mapped* tensor, which contains the neighboring node features of each edge.
Node features can be automatically mapped by appending :obj:`_i` or :obj:`_j` to the variable name.
In fact, any tensor can be mapped this way, as long as they have :math:`N` entries in its first dimension.

The neighboring node features are normalized by computing node degrees :math:`\deg(i)` for each node :math:`i` and saving :math:`1/(\sqrt{\deg(i)} \cdot \sqrt{\deg(j)})` in :obj:`norm` for each edge :math:`(i,j) \in \mathcal{E}`.

In the :meth:`update` function, we simply return the output of the aggregation.

这就是创建简单的消息传递层所需的全部。您可以将此层用作深度架构的组件。初始化和调用很简单：

.. code-block:: python

    conv = GCNConv(16, 32)
    x = conv(x, edge_index)

边卷积层的实现
---------------------------------

`edge convolutional layer <https://arxiv.org/abs/1801.07829>`_ 用于处理图或点云，数学上定义为

.. math::

    \mathbf{x}_i^{(k)} = \max_{j \in \mathcal{N}(i)} h_{\mathbf{\Theta}} \left( \mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)} - \mathbf{x}_i^{(k-1)} \right),

其中 :math:`h_{\mathbf{\Theta}}` 表示一个 MLP. 类似与 GCN layer, 我们也可以用 :class:`torch_geometric.nn.MessagePassing` class 来实现 Edge Convolution, this time using the :obj:`"max"` aggregation:

.. code-block:: python

    import torch
    from torch.nn import Sequential as Seq, Linear, ReLU
    from torch_geometric.nn import MessagePassing

    class EdgeConv(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super(EdgeConv, self).__init__(aggr='max') #  "Max" aggregation.
            self.mlp = Seq(Linear(2 * in_channels, out_channels),
                           ReLU(),
                           Linear(out_channels, out_channels))

        def forward(self, x, edge_index):
            # x has shape [N, in_channels]
            # edge_index has shape [2, E]

            return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

        def message(self, x_i, x_j):
            # x_i has shape [E, in_channels]
            # x_j has shape [E, in_channels]

            tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
            return self.mlp(tmp)

        def update(self, aggr_out):
            # aggr_out has shape [N, out_channels]

            return aggr_out

Inside the :meth:`message` function, we use :obj:`self.mlp` to transform both the target node features :obj:`x_i` and the relative source node features :obj:`x_j - x_i` for each edge :math:`(j,i) \in \mathcal{E}`.

边卷积实际上是一种动态卷积，它使用特征空间中的最近邻居重新计算每一层的图。幸运的是，PyTorch Geometric comes with a GPU accelerated batch-wise k-NN graph generation method named :meth:`torch_geometric.nn.knn_graph`:


.. code-block:: python

    from torch_geometric.nn import knn_graph

    class DynamicEdgeConv(EdgeConv):
        def __init__(self, in_channels, out_channels, k=6):
            super(DynamicEdgeConv, self).__init__(in_channels, out_channels)
            self.k = k

        def forward(self, x, batch=None):
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
            return super(DynamicEdgeConv, self).forward(x, edge_index)

Here, :meth:`knn_graph` computes a nearest neighbor graph, which is further used to call the :meth:`forward` method of :class:`EdgeConv`.

这为我们提供了一个干净的接口，用于初始化和调用此层：

.. code-block:: python

    conv = DynamicEdgeConv(3, 128, k=6)
    x = conv(pos, batch)
