import pandas as pd
from mygene import MyGeneInfo
import anndata as ad
import csv
import numpy as np

# Define key immune-related genes
key_genes = [
    "TRAV", "TRBV", "TRAJ", "TRBJ", "TRAC", "TRBC", "TRBD", "IGHV", "IGKV", "IGHD", "IGHJ", "IGHC",
    "CD3D", "CD3E", "CD4", "CD8A", "CD19", "MS4A1", "CD79A", "CD79B", "NKG7", "KLRD1", "CD14",
    "CD68", "ITGAX", "CD69", "IL2RA", "PDCD1", "CTLA4", "IL2", "IL10", "IFNG", "TNF"
]

def clean_gene_id(gene_id: str) -> str:
    """Strip version number from gene IDs."""
    if '.' in gene_id:
        return gene_id.split('.')[0]
    return gene_id

def resolve_with_mygene(gene_ids, batch_size=1000):
    """
    Resolve gene IDs using MyGene.info API.
    """
    mg = MyGeneInfo()
    resolved_data = []

    print(f"Resolving {len(gene_ids)} gene IDs using MyGene.info...")
    for i in range(0, len(gene_ids), batch_size):
        batch = gene_ids[i:i + batch_size]
        batch = [clean_gene_id(g) for g in batch]

        try:
            results = mg.querymany(batch, scopes="ensembl.gene", fields="symbol", species="human")
            for result in results:
                original_id = result.get("query", "")
                symbol = result.get("symbol", "Unresolved")
                resolved_data.append({"gene_id": original_id, "symbol": symbol})
        except Exception as e:
            print(f"Error resolving batch {i}-{i+batch_size}: {e}")

    return pd.DataFrame(resolved_data)

def filter_for_key_genes(resolved_df):
    """
    Filter resolved gene IDs to keep only key genes.
    """
    filtered_df = resolved_df[resolved_df["symbol"].isin(key_genes)]
    return filtered_df

def create_output_file(filtered_df, output_file):
    """
    Save filtered genes to a CSV file.
    """
    try:
        filtered_df.to_csv(output_file, index=False)
        print(f"Filtered gene data saved to {output_file}")
    except Exception as e:
        print(f"Error saving filtered data to {output_file}: {e}")

def create_gene_expression_matrix(adata, filtered_df, output_file, n_samples=1000):
    """
    Create a DataFrame with donors as rows and key genes as columns.
    Following the approach from Preprocess.py to avoid memory issues.
    """
    try:
        print(f"Processing {n_samples} donors...")
        
        # Get subset of data in memory (following Preprocess.py pattern)
        adata_subset = adata[:n_samples].to_memory()
        
        # Convert to DataFrame directly
        expression_df = adata_subset.to_df()
        
        # Get the gene indices we want to keep
        key_gene_ids = filtered_df['gene_id'].values
        columns_to_keep = [col for col in expression_df.columns if col in key_gene_ids]
        
        # Subset the DataFrame to only include key genes
        filtered_expression = expression_df[columns_to_keep]
        
        # Rename columns from gene IDs to gene symbols
        gene_id_to_symbol = dict(zip(filtered_df['gene_id'], filtered_df['symbol']))
        filtered_expression.columns = [gene_id_to_symbol[col] for col in filtered_expression.columns]
        
        # Save to CSV
        print(f"Saving expression matrix to {output_file}...")
        filtered_expression.to_csv(output_file)
        print("Done!")
        
        return filtered_expression
        
    except Exception as e:
        print(f"Error creating expression matrix: {e}")
        return None

if __name__ == "__main__":
    # File paths
    h5ad_file_path = "d:\\shawn\\Documents\\GitHub\\ResearchProject\\datav2.h5ad"
    output_file_path = "donor_key_genes_expression.csv"
    
    print("Loading .h5ad file in backed mode...")
    adata = ad.read_h5ad(h5ad_file_path, backed="r")
    
    # Extract gene IDs from the .h5ad data
    gene_ids = list(adata.var_names)
    print(f"Extracted {len(gene_ids)} gene IDs from .h5ad file.")
    
    # Resolve gene IDs
    resolved_df = resolve_with_mygene(gene_ids)
    
    # Filter for key genes
    filtered_df = filter_for_key_genes(resolved_df)
    print(f"Found {len(filtered_df)} key genes.")
    
    # Create and save the expression matrix
    result_df = create_gene_expression_matrix(adata, filtered_df, output_file_path)
    
    if result_df is not None:
        print("\nFirst few rows of the processed data:")
        print(result_df.head())
        print("\nShape of the processed data:", result_df.shape)
