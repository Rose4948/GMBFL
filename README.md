# GMBFL
> GMBFL: An improve Mutation-Based Fault Localization with the aid of Graph representation learning and graph neural work.

## Introduction
* This project corresponds to the paper `GMBFL: Revolutionizing Mutation-Based Fault Localization with Graph Representations` (doi links will be added in the future).


## Environment
PyTorch: V1.13.0  

OS: CentOS Linux release 7.9.2009 (Core)  


## Using GMBFL
> Example commands  

Find Faults for a specific project (commons lang):  

* python runtotal.py Lang 0 0.01 60 GGANN 15 3  

where runtotal.py is main entry file. Using the above command, GMBFL would execute the run.py, GGANN.py, sum.py, respectively.    


> Note  

* The third, fourth, fifth, seventh and eighth parameters in the `Example commands` are `random seed`, `learning rate`, `batch size`, `training epoch`, `number of model layers`, respectively.  

* These values in the `Example commands` all are default configuration on GMBFL. If you are making a first attempt at using GMBFL in your project, it is recommended to use the default parameters.  

* `GGANN` specifies the use of gated graph attention neural network models. Since the adjacency matrix representing the graph structure is a sparse matrix, we additionally provide a sparse matrix-based gated graph attention neural network `SpGGANN` to reduce the space required at runtime. If you do, configure it in `Transformer.py`.  

    
>  Configuration of Multi-head attention  

Due to space limitations, we did not use the multi-head attention mechanism. However, the advantages of multi-head attention have been generally acknowledged, and we also retain the interface to configure the multi-head attention mechanism. If you do, go to `model.py`.  

> Important Files  

* `runtotal.py` receives the parameters entered by the user and performs the experiment for the user specified project according to the corresponding configuration.

* `run.py` is responsible for training the ranking model for each buggy version of the project under test and predicting the fault location based on its graph representation. Each version of the ranking results is saved in a separate pkl file.

* `GGANN.py` provides detailed code for both `GGANN` and `SpGGANN`.  Please choose which one to use according to your actual situation.

* `sum.py` merges the results for all the buggy version of the project under test and  stores them in a pkl file.
 In addition, metrics about `top-1`,`top-3`, and `top-5` are displayed in the console.  

> Data  
data link [link]([https://markdown.com.cn](https://www.aliyundrive.com/s/zh7dgQqAsbn) "GMBFL-data-pkl and password:`3dz5`") 
