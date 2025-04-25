import pandas as pd
from mygene import MyGeneInfo
import anndata as ad
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric.nn import DeepGCNLayer, GENConv, LayerNorm
from torch_geometric.explain import Explainer, GNNExplainer

# --- CONFIGURATION --- #
key_genes = [
    "TRAV", "TRBV", "TRAJ", "TRBJ", "TRAC", "TRBC", "TRBD", "IGHV", "IGKV", "IGHD", "IGHJ", "IGHC",
    "CD3D", "CD3E", "CD4", "CD8A", "CD19", "MS4A1", "CD79A", "CD79B", "NKG7", "KLRD1", "CD14",
    "CD68", "ITGAX", "CD69", "IL2RA", "PDCD1", "CTLA4", "IL2", "IL10", "IFNG", "TNF"
]

train_file = "d:/shawn/Documents/GitHub/ResearchProject/datav2.h5ad"
test_file = "d:/shawn/Documents/GitHub/ResearchProject/cancer_data.h5ad"

sample_size = 100

# --- FUNCTIONS --- #
def clean_gene_id(gene_id):
    return gene_id.split(".")[0] if "." in gene_id else gene_id

def resolve_gene_ids(gene_ids):
    mg = MyGeneInfo()
    cleaned = [clean_gene_id(gid) for gid in gene_ids]
    result = mg.querymany(cleaned, scopes="ensembl.gene", fields="symbol", species="human")
    return pd.DataFrame([{ "gene_id": r.get("query"), "symbol": r.get("symbol", "Unresolved") } for r in result])

def filter_key_genes(df):
    return df[df["symbol"].isin(key_genes)]

def load_filtered_expression(h5ad_path, resolved_df, sample_size):
    """
    Load gene expression data using backed mode efficiently.
    Filters for key genes based on resolved gene IDs and loads only a subset of data.
    """
    try:
        print("Loading .h5ad in backed mode...")
        adata = ad.read_h5ad(h5ad_path, backed="r")

        # Map gene IDs to symbols
        key_gene_ids = resolved_df['gene_id'].values
        gene_id_to_symbol = dict(zip(resolved_df['gene_id'], resolved_df['symbol']))

        # Select genes that exist in the dataset
        selected_gene_ids = [gene for gene in adata.var_names if gene in key_gene_ids]
        if not selected_gene_ids:
            raise ValueError("No matching gene IDs found in the dataset.")
        
        print(f"Found {len(selected_gene_ids)} matching genes.")

        # Convert backed mode data into memory-efficient processing
        print(f"Loading expression data for first {sample_size} donors...")
        adata_subset = adata[:sample_size].to_memory()  # Move subset into memory
        
        # Convert to DataFrame while avoiding excessive memory usage
        expr_df = adata_subset.to_df()

        # Subset only the key genes
        columns_to_keep = [col for col in expr_df.columns if col in selected_gene_ids]
        filtered_expression = expr_df[columns_to_keep]

        # Rename columns from gene IDs to gene symbols
        filtered_expression.columns = [gene_id_to_symbol[col] for col in filtered_expression.columns]

        print("Expression data successfully loaded.")
        return filtered_expression

    except Exception as e:
        print(f"Error during data loading: {e}")
        return None



def create_graph_data(expr_df):
    similarity_matrix = cosine_similarity(expr_df)
    graph = nx.Graph()

    for i in range(len(expr_df)):
        for j in range(i + 1, len(expr_df)):
            similarity = similarity_matrix[i, j]
            if similarity > 0.5:
                graph.add_edge(i, j, weight=similarity)

    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    x = torch.tensor(expr_df.values, dtype=torch.float)
    y = torch.tensor(np.random.randint(0, 2, size=len(expr_df)), dtype=torch.long)  # Placeholder labels

    return Data(x=x, edge_index=edge_index, y=y)

def compute_jsd(p, q):
    p = np.array(p) / np.sum(p)
    q = np.array(q) / np.sum(q)
    return jensenshannon(p, q)

class BayesianDeeperGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.node_encoder = torch.nn.Linear(in_channels, hidden_channels)
        self.layers = torch.nn.ModuleList()
        for _ in range(3):
            conv = GENConv(hidden_channels, hidden_channels)
            norm = LayerNorm(hidden_channels, affine=True)
            act = torch.nn.ReLU()
            self.layers.append(DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1))
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.node_encoder(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.classifier(x)

# --- Risk Prediction --- #
def get_predictions(model, data):
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index).squeeze()
        return preds[:, 1].cpu().numpy()  # Assuming index 1 is the probability of "risky"

def calculate_risk_threshold(preds, method="mean+std"):
    if method == "mean+std":
        return preds.mean() + preds.std()
    elif method == "percentile":
        return np.percentile(preds, 95)
    else:
        raise ValueError("Invalid threshold method selected.")

def analyze_internal_risks(model, data, sample_names=None):
    preds = get_predictions(model, data)
    threshold = calculate_risk_threshold(preds)

    print("\n--- Internal Risk Analysis ---")
    for i, pred in enumerate(preds):
        flag = "⚠️ POSSIBLE RISK" if pred > threshold else ""
        name = sample_names[i] if sample_names else f"Sample_{i}"
        print(f"{name}: Predicted Risk Score = {pred:.4f} {flag}")
    
    return preds, threshold

# --- MAIN EXECUTION --- #
if __name__ == "__main__":
    print("Loading training data and resolving gene symbols...")
    train_adata = ad.read_h5ad(train_file, backed="r")
    gene_ids = list(train_adata.var_names)
    resolved = resolve_gene_ids(gene_ids)
    filtered_genes = filter_key_genes(resolved)

    print("Creating expression matrices for train and test datasets...")
    train_expr = load_filtered_expression(train_file, filtered_genes, sample_size)
    test_expr = load_filtered_expression(test_file, filtered_genes, sample_size)

    # Ensure both datasets contain the same genes
    common_genes = list(set(train_expr.columns) & set(test_expr.columns))

    # Filter both DataFrames to use only these common genes
    train_expr = train_expr[common_genes]
    test_expr = test_expr[common_genes]

    print("Computing Jensen-Shannon Divergence between train and test...")
    jsd = compute_jsd(train_expr.mean(axis=0), test_expr.mean(axis=0))
    print("JSD between train and test distributions:", jsd)

    print("Creating graph structures for GCN...")
    train_data = create_graph_data(train_expr)
    test_data = create_graph_data(test_expr)

    print("Training BayesianDeeperGCN...")
    model = BayesianDeeperGCN(train_data.num_node_features, 64, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index)
        loss = F.cross_entropy(out, train_data.y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(),
    explanation_type='model',
    edge_mask_type='object',
    node_mask_type='attributes',
    model_config={"mode": "binary_classification", "task_level": "node", "return_type": "raw"}  # Add 'task_level'
)

    explanation = explainer(test_data.x, edge_index=test_data.edge_index)
    print("Top influential features:", explanation.node_mask)

    # --- RISK PREDICTION ON TRAINING DATA ---
    print("Running internal risk analysis on training data...")
    preds, threshold = analyze_internal_risks(model, train_data)

    print("Explaining top risky samples using GNNExplainer...")
    risky_indices = np.where(preds > threshold)[0]

    for idx in risky_indices[:3]:  # Just show top 3 for now
        explanation = explainer(x=test_data.x, edge_index=test_data.edge_index)
    print(f"\nExplanation for node {idx}:")
    print("Top influential features:", explanation.node_mask)


    print("Pipeline complete.")
