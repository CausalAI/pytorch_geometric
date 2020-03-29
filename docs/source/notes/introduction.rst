PyG中基本概念
=======================
 

我们将简要介绍 PyTorch Geometric 的基本概念 through self-contained examples.
At its core, PyTorch Geometric 提供以下主要功能：

.. contents::
    :local:

图数据类
-----------------------

图被用于 model pairwise relations (edges) between objects (nodes).
PyTorch Geometric中的图由 :class:`torch_geometric.data.Data` 的实例来描述，该实例默认情况下具有以下属性：

- ``data.x``: 节点特征矩阵 with shape ``[num_nodes, num_node_features]``
- ``data.edge_index``: COO 格式的图连接结构 with shape ``[2, num_edges]`` and 数据类型 ``torch.long``
- ``data.edge_attr``: 边特征矩阵 with shape ``[num_edges, num_edge_features]``
- ``data.y``: Target to train against (may have arbitrary shape), *e.g.*, node-level targets of shape ``[num_nodes, *]`` or graph-level targets of shape ``[1, *]``
- ``data.pos``: 节点位置矩阵 with shape ``[num_nodes, num_dimensions]``

这些属性都不是必需的。 实际上， the ``Data`` object is not even restricted to these attributes. 例如, We can extend it by ``data.face`` to save the connectivity of triangles from a 3D mesh in a tensor with shape ``[3, num_faces]`` and type ``torch.long``.

.. Note::
    PyTorch and ``torchvision`` define an example as a tuple of an image and a target. 我们在 PyTorch Geometric 中放弃了这种表示法 to allow for various data structures in a clean and understandable way.

我们展示一个简单的例子 of an unweighted and undirected graph with three nodes and four edges. 每个节点包含一个特征:

.. code-block:: python

    import torch
    from torch_geometric.data import Data

    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    >>> Data(edge_index=[2, 4], x=[3, 1])

.. image:: ../_figures/graph.svg
  :align: center
  :width: 300px

请注意，``edge_index``，即定义所有边的源节点和目标节点的张量，它不是索引元组的列表。
如果要以这种方式编写索引，则应在将索引传递给数据构造函数之前，对其转置并调用 ``contiguous`` 方法：


.. code-block:: python

    import torch
    from torch_geometric.data import Data

    edge_index = torch.tensor([[0, 1],
                               [1, 0],
                               [1, 2],
                               [2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index.t().contiguous())
    >>> Data(edge_index=[2, 4], x=[3, 1])

尽管图只有两条边，但是我们需要定义四个索引元组以说明边的两个方向。

.. Note::
    您可以随时打印出数据对象，并获得有关其属性及其形状的简短信息。

Besides of being a plain old python object, :class:`torch_geometric.data.Data` provides a number of utility functions, *e.g.*:

.. code-block:: python

    print(data.keys)
    >>> ['x', 'edge_index']

    print(data['x'])
    >>> tensor([[-1.0],
                [0.0],
                [1.0]])

    for key, item in data:
        print("{} found in data".format(key))
    >>> x found in data
    >>> edge_index found in data

    'edge_attr' in data
    >>> False

    data.num_nodes
    >>> 3

    data.num_edges
    >>> 4

    data.num_node_features
    >>> 1

    data.contains_isolated_nodes()
    >>> False

    data.contains_self_loops()
    >>> False

    data.is_directed()
    >>> False

    # Transfer data object to GPU.
    device = torch.device('cuda')
    data = data.to(device)

You can find a complete list of all methods at :class:`torch_geometric.data.Data`.

基准数据集
-------------------------

PyTorch Geometric 包含大量常见基准数据集，例如所有 Planetoid 数据集（Cora，Citeseer，Pubmed）all graph classification datasets from `http://graphkernels.cs.tu-dortmund.de/ <http://graphkernels.cs.tu-dortmund.de/>`_ and their `cleaned versions <https://github.com/nd7141/graph_datasets>`_, QM7和QM9数据集, and a handful of 3D mesh/point cloud datasets like FAUST, ModelNet10/40 and ShapeNet.

初始化数据集很简单。数据集的初始化将自动下载其原始文件，并将其处理为先前描述的 ``Data`` 格式。
例如，要加载 ENZYMES 数据集(consisting of 600 graphs within 6 classes)，请输入：

.. code-block:: python

    from torch_geometric.datasets import TUDataset

    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    >>> ENZYMES(600)

    len(dataset)
    >>> 600

    dataset.num_classes
    >>> 6

    dataset.num_node_features
    >>> 3

现在，我们可以访问数据集中的所有600个图：

.. code-block:: python

    data = dataset[0]
    >>> Data(edge_index=[2, 168], x=[37, 3], y=[1])

    data.is_undirected()
    >>> True


我们可以看到数据集中的第一个图包含37个节点，每个节点都有3个特征。有168/2 = 84条无向边，并且该图is assigned to exactly one class. In addition, the data object is holding exactly one graph-level target.


(分割数据集) We can even use slices, long or byte tensors to split the dataset.
*E.g.*, to create a 90/10 train/test split, type:

.. code-block:: python

    train_dataset = dataset[:540]
    >>> ENZYMES(540)

    test_dataset = dataset[540:]
    >>> ENZYMES(60)

If you are unsure whether the dataset is already shuffled before you split, you can randomly permutate it by running:

.. code-block:: python

    dataset = dataset.shuffle()
    >>> ENZYMES(600)

这个相当于如下：

.. code-block:: python

    perm = torch.randperm(len(dataset))
    dataset = dataset[perm]
    >> ENZYMES(600)

让我们再尝试另外一个数据集 Cora，这是用于半监督图节点分类的标准基准数据集：

.. code-block:: python

    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    >>> Cora()

    len(dataset)
    >>> 1

    dataset.num_classes
    >>> 7

    dataset.num_node_features
    >>> 1433

Here, the dataset contains only a single, undirected citation graph:

.. code-block:: python

    data = dataset[0]
    >>> Data(edge_index=[2, 10556], test_mask=[2708],
             train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])

    data.is_undirected()
    >>> True

    data.train_mask.sum().item()
    >>> 140

    data.val_mask.sum().item()
    >>> 500

    data.test_mask.sum().item()
    >>> 1000

This time, the ``Data`` objects holds a label for each node, and additional attributes: ``train_mask``, ``val_mask`` and ``test_mask``:

- ``train_mask`` denotes against which nodes to train (140 nodes)
- ``val_mask`` denotes which nodes to use for validation, *e.g.*, to perform early stopping (500 nodes)
- ``test_mask`` denotes against which nodes to test (1000 nodes)

Mini-batches
------------

(Networks 的批量化并行训练) Neural networks are usually trained in a batch-wise fashion. PyTorch Geometric achieves parallelization over a mini-batch by creating sparse block diagonal adjacency matrices (defined by ``edge_index`` and ``edge_attr``) and concatenating feature and target matrices in the node dimension. This composition allows differing number of nodes and edges over examples in one batch:

.. math::

    \mathbf{A} = \begin{bmatrix} \mathbf{A}_1 & & \\ & \ddots & \\ & & \mathbf{A}_n \end{bmatrix}, \qquad \mathbf{X} = \begin{bmatrix} \mathbf{X}_1 \\ \vdots \\ \mathbf{X}_n \end{bmatrix}, \qquad \mathbf{Y} = \begin{bmatrix} \mathbf{Y}_1 \\ \vdots \\ \mathbf{Y}_n \end{bmatrix}

(数据的批量载入) PyTorch Geometric contains its own :class:`torch_geometric.data.DataLoader`, which already takes care of this concatenation process.
Let's learn about it in an example:

.. code-block:: python

    from torch_geometric.datasets import TUDataset
    from torch_geometric.data import DataLoader

    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in loader:
        batch
        >>> Batch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])

        batch.num_graphs
        >>> 32

:class:`torch_geometric.data.Batch` 继承自 :class:`torch_geometric.data.Data` 并包含额外的属性 ``batch``。

``batch`` is a column vector which maps each node to its respective graph in the batch:

.. math::

    \mathrm{batch} = {\begin{bmatrix} 0 & \cdots & 0 & 1 & \cdots & n - 2 & n -1 & \cdots & n - 1 \end{bmatrix}}^{\top}

You can use it to, *e.g.*, average node features in the node dimension for each graph individually:

.. code-block:: python

    from torch_scatter import scatter_mean
    from torch_geometric.datasets import TUDataset
    from torch_geometric.data import DataLoader

    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for data in loader:
        data
        >>> Batch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])

        data.num_graphs
        >>> 32

        x = scatter_mean(data.x, data.batch, dim=0)
        x.size()
        >>> torch.Size([32, 21])

您可以了解有关 PyTorch Geometric 的内部批处理程序的更多信息, *e.g.*, how to modify its behaviour, `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html>`_.
For documentation of scatter operations, we refer the interested reader to the ``torch-scatter`` `documentation <https://pytorch-scatter.readthedocs.io>`_.

Data Transforms
---------------

Transforms are a common way in ``torchvision`` to transform images and perform augmentation. PyTorch Geometric带有自己的 transforms，将 ``Data`` 对象作为输入并返回一个新的变换后的 ``Data`` 对象。 Transforms can be chained together using :class:`torch_geometric.transforms.Compose` and are applied before saving a processed dataset on disk (``pre_transform``) or before accessing a graph in a dataset (``transform``).

让我们看一个示例，在该示例上，我们对 ShapeNet 数据集上应用 transforms(containing 17,000 3D shape point clouds and per point labels from 16 shape categories)。


.. code-block:: python

    from torch_geometric.datasets import ShapeNet

    dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])

    dataset[0]
    >>> Data(pos=[2518, 3], y=[2518])

We can convert the point cloud dataset into a graph dataset by generating nearest neighbor graphs from the point clouds via transforms:

.. code-block:: python

    import torch_geometric.transforms as T
    from torch_geometric.datasets import ShapeNet

    dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
                        pre_transform=T.KNNGraph(k=6))

    dataset[0]
    >>> Data(edge_index=[2, 15108], pos=[2518, 3], y=[2518])

.. note::
    We use the ``pre_transform`` to convert the data before saving it to disk (从而缩短了加载时间).
    Note that the next time the dataset is initialized it will already contain graph edges, even if you do not pass any transform.

In addition, we can use the ``transform`` argument to randomly augment a ``Data`` object, *e.g.* translating each node position by a small number:

.. code-block:: python

    import torch_geometric.transforms as T
    from torch_geometric.datasets import ShapeNet

    dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
                        pre_transform=T.KNNGraph(k=6),
                        transform=T.RandomTranslate(0.01))

    dataset[0]
    >>> Data(edge_index=[2, 15108], pos=[2518, 3], y=[2518])

You can find a complete list of all implemented transforms at :mod:`torch_geometric.transforms`.

图网络端对端例子
--------------------------

在了解了PyTorch Geometric中的数据处理，数据集，loader and transforms 之后，是时候实现我们的第一个图神经网络了！ We will use a simple GCN layer and replicate the experiments on the Cora citation dataset. 有关 GCN 的更深解释, 请参考 `blog post <http://tkipf.github.io/graph-convolutional-networks/>`_.

我们首先需要加载Cora数据集：

.. code-block:: python

    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    >>> Cora()

Note that we do not need to use transforms or a dataloader.
Now let's implement a two-layer GCN:

.. code-block:: python

    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(dataset.num_node_features, 16)
            self.conv2 = GCNConv(16, dataset.num_classes)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)

            return F.log_softmax(x, dim=1)

The constructor defines two ``GCNConv`` layers which get called in the forward pass of our network. Note that the non-linearity is not integrated in the ``conv`` calls and hence needs to be applied afterwards (something which is consistent accross all operators in PyTorch Geometric). Here, we chose to use ReLU as our intermediate non-linearity between and finally output a softmax distribution over the number of classes.

我们在 train nodes 上训练此模型 200 个 epochs：

.. code-block:: python

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

最后，我们可以在 test nodes 上评估模型：

.. code-block:: python

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    print('Accuracy: {:.4f}'.format(acc))
    >>> Accuracy: 0.8150

这就是实现您的第一个图神经网络所需要的。了解图卷积和池化的最简单方法是 to study the examples in the ``examples/`` directory and to browse :mod:`torch_geometric.nn`.
Happy hacking!
