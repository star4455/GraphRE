Understanding the Robustness of Graph Neural Networks against Adversarial Attacks

Datasets

(1) Cora is a citation network with 2,485 papers in 7 classes, represented by 1,433-dimensional binary word vectors; 
(2) Citeseer is a citation network with 2,110 papers in 6 classes and 3,668 citation links, represented by binary word vector features; 
(3) PubMed is a citation network of 19,717 diabetes-related papers in 3 classes, connected by 44,338 links and represented by 500-dimensional TF-IDF vectors; 
(4) Amazon Photo is a co-purchase network of 7,650 products in 8 categories, represented by 745-dimensional bag-of-words features and connected through 238,162 co-purchase edges.

Attacks

Regarding adversarial attacks, we adopt three widely used methods: Mettack, Nettack, and Random Attack. For Mettack and Random Attack, we apply structural perturbations with perturbation rates ranging from 2% to 10% (i.e., 2%, 4%, 5%, 6%, 8%, and 10%). For Nettack, we perturb each target node with either 2.0 or 4.0 structural modifications. 

Parameter Settings

All experiments are conducted over ten independent runs, and the reported results are averaged accordingly to ensure statistical reliability. The models are implemented in PyTorch version 1.8.2 and executed on an NVIDIA RTX 3080Ti GPU equipped with CUDA 11.1. The ReLU activation function is used throughout, the dropout rate is set to 0.5 to prevent overfitting, and all models are consistently trained for a total of 200 epochs.
