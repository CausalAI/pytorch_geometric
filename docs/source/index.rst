:github_url: https://causalai.github.io/pytorch_geometric/

PyTorch Geometric Documentation
===============================

本 `项目 <https://github.com/CausalAI/pytorch_geometric>`_ 是关于 PyG 的编译和解读项目 by Heyang Gong; 大概会包含如下的内容:

1. 简单翻译和理解 PyG 的官方教程
2. 使用 PyG 来理解和实现图网络的基本内容

PyTorch Geometric is a geometric deep learning extension library for `PyTorch <https://pytorch.org/>`_.

It consists of various methods for deep learning on graphs and other irregular structures, also known as `geometric deep learning <http://geometricdeeplearning.com/>`_, from a variety of published papers.
In addition, it consists of an easy-to-use mini-batch loader, a large number of common benchmark datasets (based on simple interfaces to create your own), and helpful transforms, both for learning on arbitrary graphs as well as on 3D meshes or point clouds.

.. toctree:: 
   :glob:
   :maxdepth: 1
   :caption: Notes

   notes/installation
   notes/introduction
   notes/create_gnn
   notes/create_dataset
   notes/batching
   notes/resources

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Graph Networks

   notebooks/01-quick_survey
   notebooks/02-researchers
   
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference

   modules/root
   modules/nn
   modules/data
   modules/datasets
   modules/transforms
   modules/utils
   modules/io

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
