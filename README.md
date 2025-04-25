Gene Expression Risk Prediction Using Bayesian Deep Graph Convolutional Networks (GCNs)

Project Overview

This project focuses on predicting risk scores for patients based on their gene expression profiles using a Bayesian Deeper Graph Convolutional Network (GCN). It leverages machine learning to analyze gene expression similarities among patients and flag potential risks based on model predictions. Instead of using Jensen-Shannon Divergence (JSD) to construct the graph, the pipeline builds relationships using cosine similarity, ensuring that patients with similar gene expression profiles are connected within the graph structure. JSD is still calculated but serves as a measure of distribution shifts between datasets rather than determining edge weights.

To achieve this, the pipeline begisn with gene ID resolution, mapping gene identifiers to official symbols using the mygene API. After filtering for key genes related to immune function and cancer, the model extracts gene expression matrices from .h5ad files while ensuring only relevant features are retained across datasets. The graph is constructed by computing cosine similarity between gene expression profiles, converting this structure into a format compatible with torch_geometric for training the GCN.

The model is trained using DeeperGCN layers with GENConx to learn meaningful patterns in thed ata. It predicts risk scores, which are analyzed against a threshold derived from either the mean + standard deviation or the 95th percentile of scores. Samples exceeding the threshold are flagged as potential risks, providing insights into patients who may require further investigation.

To enhance model interpretability, this project employs GNNExplainer, which identifies influential feature driving predictions. The most impactful gene contributing to risk scores are visualized using bar plots, helping to udnerstand the biological markers assocaited with risk. This approach ensures the predictions are not just blackbox outputs but are explainable.

Project Structure:
The repository contains the following mains ections:
1. Data Preprocessing: Clean and filter the gene expression dataset, handle missing and zero values, and normalize the data.
2. Jensen-Shannon Divergence (JSD) Calculation: Compute the pairwise JSD between samples to build an adjacency matrix for graph-based modeling.
3. Graph Construction: Create a graph from the adjacency matrix, with gene expression profiles as node features and JSD as edge weights.
4. Modeling with GCN: Implement a Bayesian Deeper Graph Convolutional Network to predict risk scores for each patient.
5. Risk Analysis: Perform internal risk analysis by comparing predicted scores with calculated thresholds.
6. Interpretability: Use Integrated Gradients to interpret the model’s predictions and highlight the top contributing genes.

Files and Directories

- Preprocess/DonorKey/donor_key_genes_expression.csv: Raw gene expression data of donors (rows represent donors, columns represent genes).

- node_features.npy: Processed gene expression data with zeros replaced by epsilon and normalized to probability distributions.

- adj_matrix.npy: Adjacency matrix computed using Jensen-Shannon Divergence between donors.

- labels.npy: Labels for the donors, with "0" indicating healthy donors.

- jsd_graph.pkl: Pickled graph object containing donor relationships.

- model.py: Code for building and training the Bayesian Deeper GCN.

- train.py: Code for model training, risk analysis, and predictions.

Dependencies

The following Python libraries are required:
- Pandas, Numpy, Scipy
- Torch, Torch_geometric
- Networkx, Matplotlib
- Anndata
- Mygene

Data
- Train file: datav2.h5ad
- Test file: cancer_data.h5ad
- Key genes: A predefined list of relevant genes

Pipeline Steps
1. Gene ID Resolution
- Gene IDs are cleaned and ampped to official symbols using mygene API.
2. Expression Matrix Filtering
- .h5ad files are read to extract gene expression values
3. Jensen-Shannon Divergence Calculation
- Measures the similarity between the mean expression distributions of training and testing datasets.
4. Graph Construction
- Computes cosine similarity between samples
- Constructs a weighted graph where edges represent similarity between samples
5. Bayesian DeeperGCN Model
- Use multiple DeepGCN layers with GENConv and LayerNorm.
- Trained using CrossEntropy loss and Adam optimizer
6. Risk Analysis
- Predictions are obtained from the trained model
7. Model Explainability
- GNNEXplainer identifies influential features affecting predictions.

Example Outputs:
--- Internal Risk Analysis ---
Sample_0: Predicted Risk Score = 0.0014
Sample_1: Predicted Risk Score = 0.0145
Sample_2: Predicted Risk Score = 0.1712
Sample_3: Predicted Risk Score = 0.4099
Sample_4: Predicted Risk Score = 0.1440
Sample_5: Predicted Risk Score = 0.0169
Sample_6: Predicted Risk Score = 0.0169
Sample_7: Predicted Risk Score = 0.9986 ⚠️ POSSIBLE RISK
Sample_8: Predicted Risk Score = 0.1587
Sample_9: Predicted Risk Score = 0.0013
Sample_10: Predicted Risk Score = 0.3140
Sample_11: Predicted Risk Score = 0.0641
Sample_12: Predicted Risk Score = 0.3777
Sample_13: Predicted Risk Score = 0.1312
Sample_14: Predicted Risk Score = 0.6595 ⚠️ POSSIBLE RISK
Sample_15: Predicted Risk Score = 0.1893
Sample_16: Predicted Risk Score = 0.3198
Sample_17: Predicted Risk Score = 0.1419
Sample_18: Predicted Risk Score = 0.8296 ⚠️ POSSIBLE RISK
Sample_19: Predicted Risk Score = 0.2446
Sample_20: Predicted Risk Score = 0.8861 ⚠️ POSSIBLE RISK
Sample_21: Predicted Risk Score = 0.2701
Sample_22: Predicted Risk Score = 0.1793
Sample_23: Predicted Risk Score = 0.1828
Sample_24: Predicted Risk Score = 0.0584
Sample_25: Predicted Risk Score = 0.0731
Sample_26: Predicted Risk Score = 0.2145
Sample_27: Predicted Risk Score = 0.4297
Sample_28: Predicted Risk Score = 0.0635
Sample_29: Predicted Risk Score = 0.0232
Sample_30: Predicted Risk Score = 0.8617 ⚠️ POSSIBLE RISK
Sample_31: Predicted Risk Score = 0.5570
Sample_32: Predicted Risk Score = 0.1701
Sample_33: Predicted Risk Score = 0.2033
Sample_34: Predicted Risk Score = 0.1207
Sample_35: Predicted Risk Score = 0.2294
Sample_36: Predicted Risk Score = 0.0348
Sample_37: Predicted Risk Score = 0.1672
Sample_38: Predicted Risk Score = 0.1790
Sample_39: Predicted Risk Score = 0.8229 ⚠️ POSSIBLE RISK
Sample_40: Predicted Risk Score = 0.9822 ⚠️ POSSIBLE RISK
Sample_41: Predicted Risk Score = 0.0422
Sample_42: Predicted Risk Score = 0.4405
Sample_43: Predicted Risk Score = 0.3461
Sample_44: Predicted Risk Score = 0.0864
Sample_45: Predicted Risk Score = 0.0226
Sample_46: Predicted Risk Score = 0.0613
Sample_47: Predicted Risk Score = 0.6285
Sample_48: Predicted Risk Score = 0.2244
Sample_49: Predicted Risk Score = 0.1405
Sample_50: Predicted Risk Score = 0.0389
Sample_51: Predicted Risk Score = 0.1886
Sample_52: Predicted Risk Score = 0.3817
Sample_53: Predicted Risk Score = 0.0593
Sample_54: Predicted Risk Score = 0.2383
Sample_55: Predicted Risk Score = 0.4194
Sample_56: Predicted Risk Score = 0.0927
Sample_57: Predicted Risk Score = 0.2902
Sample_58: Predicted Risk Score = 0.0206
Sample_59: Predicted Risk Score = 0.2464
Sample_60: Predicted Risk Score = 0.6925 ⚠️ POSSIBLE RISK
Sample_61: Predicted Risk Score = 0.3108
Sample_62: Predicted Risk Score = 0.3614
Sample_63: Predicted Risk Score = 0.1409
Sample_64: Predicted Risk Score = 0.2511
Sample_65: Predicted Risk Score = 0.1062
Sample_66: Predicted Risk Score = 0.5864
Sample_67: Predicted Risk Score = 0.9784 ⚠️ POSSIBLE RISK
Sample_68: Predicted Risk Score = 0.0285 
Sample_69: Predicted Risk Score = 0.0897
Sample_70: Predicted Risk Score = 0.2474
Sample_71: Predicted Risk Score = 0.9089 ⚠️ POSSIBLE RISK
Sample_72: Predicted Risk Score = 0.9953 ⚠️ POSSIBLE RISK
Sample_73: Predicted Risk Score = 0.9153 ⚠️ POSSIBLE RISK
Sample_74: Predicted Risk Score = 0.9707 ⚠️ POSSIBLE RISK
Sample_75: Predicted Risk Score = 0.5126
Sample_76: Predicted Risk Score = 0.1327
Sample_77: Predicted Risk Score = 0.1970
Sample_78: Predicted Risk Score = 0.0823
Sample_79: Predicted Risk Score = 0.0763
Sample_80: Predicted Risk Score = 0.3102
Sample_81: Predicted Risk Score = 0.2873
Sample_82: Predicted Risk Score = 0.0672
Sample_83: Predicted Risk Score = 0.4622
Sample_84: Predicted Risk Score = 0.2186
Sample_85: Predicted Risk Score = 0.2248
Sample_86: Predicted Risk Score = 0.3646
Sample_87: Predicted Risk Score = 0.8993 ⚠️ POSSIBLE RISK
Sample_88: Predicted Risk Score = 0.1741
Sample_89: Predicted Risk Score = 0.0124
Sample_90: Predicted Risk Score = 0.1319
Sample_91: Predicted Risk Score = 0.2159
Sample_92: Predicted Risk Score = 0.9262 ⚠️ POSSIBLE RISK
Sample_93: Predicted Risk Score = 0.9722 ⚠️ POSSIBLE RISK
Sample_94: Predicted Risk Score = 0.9937 ⚠️ POSSIBLE RISK
Sample_95: Predicted Risk Score = 0.8381 ⚠️ POSSIBLE RISK
Sample_97: Predicted Risk Score = 0.2460
Sample_98: Predicted Risk Score = 0.6599 ⚠️ POSSIBLE RISK
Sample_99: Predicted Risk Score = 0.0263