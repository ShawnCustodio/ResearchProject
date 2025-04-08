import pandas as pd
import requests

def parse_all_gene_list(file_path: str):
    """
    Parse the all_gene_list.txt file to extract GO and Reactome IDs.
    Args:
        file_path (str): Path to the all_gene_list.txt file.
    Returns:
        dict: Dictionary with GO and Reactome IDs grouped by source.
    """
    print(f"Parsing file: {file_path}")
    pathways = pd.read_csv(file_path, sep="\t", header=None, names=["pathway", "id", "source", "link", "gene_count"])
    go_ids = pathways[pathways["source"] == "GO"]["id"].tolist()
    reactome_ids = pathways[pathways["source"] == "Reactome"]["id"].tolist()
    print(f"Found {len(go_ids)} GO IDs and {len(reactome_ids)} Reactome IDs.")
    return {"GO": go_ids, "Reactome": reactome_ids}

def fetch_genes_from_go(go_ids: list):
    """
    Fetch genes associated with GO IDs using the Gene Ontology API.
    Args:
        go_ids (list): List of GO IDs.
    Returns:
        set: Set of gene symbols associated with the GO IDs.
    """
    print("Fetching genes from GO...")
    genes = set()
    base_url = "http://api.geneontology.org/api/bioentity/term/"
    for go_id in go_ids:
        try:
            url = f"{base_url}{go_id}/genes"
            response = requests.get(url)
            if response.ok:
                data = response.json()
                for gene in data.get("associations", []):
                    genes.add(gene["gene"]["symbol"])
        except Exception as e:
            print(f"Error fetching GO ID {go_id}: {e}")
    print(f"Retrieved {len(genes)} genes from GO.")
    return genes

def fetch_genes_from_reactome(reactome_ids: list):
    """
    Fetch genes associated with Reactome IDs using the Reactome API.
    Args:
        reactome_ids (list): List of Reactome IDs.
    Returns:
        set: Set of gene symbols associated with the Reactome IDs.
    """
    print("Fetching genes from Reactome...")
    genes = set()
    base_url = "https://reactome.org/ContentService/data/pathwayParticipants/"
    for reactome_id in reactome_ids:
        try:
            url = f"{base_url}{reactome_id}"
            response = requests.get(url)
            if response.ok:
                data = response.json()
                for participant in data:
                    if "geneName" in participant:
                        genes.add(participant["geneName"])
        except Exception as e:
            print(f"Error fetching Reactome ID {reactome_id}: {e}")
    print(f"Retrieved {len(genes)} genes from Reactome.")
    return genes

def parse_gene_sum(file_path: str):
    """
    Parse the gene-sum1.xlsx file to extract gene IDs or symbols.
    Args:
        file_path (str): Path to the gene-sum1.xlsx file.
    Returns:
        set: Set of gene symbols extracted from the file.
    """
    print(f"Parsing file: {file_path}")
    df = pd.read_excel(file_path)
    # Assuming the gene symbols are in a column named "GeneSymbol"
    if "GeneSymbol" in df.columns:
        genes = set(df["GeneSymbol"].dropna().tolist())
        print(f"Retrieved {len(genes)} genes from gene-sum1.xlsx.")
        return genes
    else:
        raise ValueError("The file does not contain a 'GeneSymbol' column.")

def save_immune_genes(genes: set, output_file: str):
    """
    Save the immune-related genes to a CSV file.
    Args:
        genes (set): Set of gene symbols.
        output_file (str): Path to save the immune-related genes.
    """
    print(f"Saving immune-related genes to {output_file}...")
    pd.DataFrame({"gene_symbol": list(genes)}).to_csv(output_file, index=False)
    print("Immune-related genes saved.")

if __name__ == "__main__":
    # Input and output file paths
    all_gene_list_file = "all_gene_lists.txt"
    gene_sum_file = "gene-sum1.xlsx"
    output_file = "immune_genes.csv"

    # Parse the all_gene_list.txt file
    ids = parse_all_gene_list(all_gene_list_file)

    # Fetch genes from GO and Reactome
    go_genes = fetch_genes_from_go(ids["GO"])
    reactome_genes = fetch_genes_from_reactome(ids["Reactome"])

    # Parse the gene-sum1.xlsx file
    gene_sum_genes = parse_gene_sum(gene_sum_file)

    # Combine all genes into a single set
    all_genes = go_genes.union(reactome_genes).union(gene_sum_genes)

    # Save the combined gene list to a CSV file
    save_immune_genes(all_genes, output_file)