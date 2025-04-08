import pandas as pd
from mygene import MyGeneInfo
from typing import List
import requests

def clean_gene_id(gene_id: str) -> str:
    """Strip version number from gene IDs"""
    if '.' in gene_id:
        return gene_id.split('.')[0]
    return gene_id

def resolve_with_mygene(gene_ids: List[str], batch_size: int = 1000) -> pd.DataFrame:
    """
    Resolve gene IDs using MyGene.info API.
    Args:
        gene_ids (List[str]): List of gene IDs to resolve.
        batch_size (int): Number of IDs to process in each batch.
    Returns:
        pd.DataFrame: DataFrame with resolved gene IDs and symbols.
    """
    mg = MyGeneInfo()
    resolved_data = []

    print(f"Resolving {len(gene_ids)} gene IDs using MyGene.info...")
    for i in range(0, len(gene_ids), batch_size):
        batch = gene_ids[i:i + batch_size]
        clean_batch = [clean_gene_id(g) for g in batch]

        try:
            results = mg.querymany(clean_batch, scopes="ensembl.gene", fields="symbol,name", species="human")
            for result in results:
                original_id = result.get("query", "")
                if "notfound" in result:
                    # If no hit is found, mark as unresolved
                    resolved_data.append({"gene_id": original_id, "symbol": "Unresolved"})
                elif "symbol" in result:
                    # If a symbol is found, use it
                    resolved_data.append({"gene_id": original_id, "symbol": result["symbol"]})
                elif "name" in result:
                    # If no symbol but a name is found, use the name
                    resolved_data.append({"gene_id": original_id, "symbol": result["name"]})
                else:
                    # If no symbol or name, mark as unresolved
                    resolved_data.append({"gene_id": original_id, "symbol": "Unresolved"})
        except Exception as e:
            print(f"Error resolving batch {i}-{i+batch_size}: {e}")

    return pd.DataFrame(resolved_data)

def resolve_with_ncbi(gene_id: str) -> str:
    """
    Resolve gene ID using NCBI E-utilities API.
    Args:
        gene_id (str): Gene ID to resolve.
    Returns:
        str: Resolved gene symbol or "Unresolved".
    """
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        search_url = f"{base_url}/esearch.fcgi?db=gene&term={gene_id}&retmode=json"
        response = requests.get(search_url)
        if response.ok:
            search_data = response.json()
            idlist = search_data.get("esearchresult", {}).get("idlist", [])
            if idlist:  # Check if idlist is not empty
                gene_id_ncbi = idlist[0]
                summary_url = f"{base_url}/esummary.fcgi?db=gene&id={gene_id_ncbi}&retmode=json"
                summary_response = requests.get(summary_url)
                if summary_response.ok:
                    summary_data = summary_response.json()
                    if "result" in summary_data and gene_id_ncbi in summary_data["result"]:
                        return summary_data["result"][gene_id_ncbi].get("name", "Unresolved")
        return "Unresolved"  # Return "Unresolved" if no results are found
    except Exception as e:
        print(f"NCBI API error for {gene_id}: {e}")
        return "Unresolved"

def resolve_gene_ids(input_file: str, resolved_file: str, unresolved_file: str, loc_file: str):
    """
    Resolve gene IDs from a text file using MyGene.info and NCBI GenBank API.
    Args:
        input_file (str): Path to the input file containing unresolved gene IDs.
        resolved_file (str): Path to save resolved gene IDs.
        unresolved_file (str): Path to save unresolved gene IDs.
        loc_file (str): Path to save IDs with "LOC" symbols.
    """
    try:
        with open(input_file, "r") as f:
            gene_ids = [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
        return

    # Step 1: Resolve with MyGene.info
    resolved_df = resolve_with_mygene(gene_ids)

    # Step 2: Resolve unresolved IDs with NCBI GenBank API
    unresolved_ids = resolved_df[resolved_df["symbol"] == "Unresolved"]["gene_id"].tolist()
    print(f"Resolving {len(unresolved_ids)} unresolved gene IDs using NCBI GenBank API...")
    for gene_id in unresolved_ids:
        resolved_symbol = resolve_with_ncbi(gene_id)
        resolved_df.loc[resolved_df["gene_id"] == gene_id, "symbol"] = resolved_symbol

    # Step 3: Separate resolved, unresolved, and "LOC" IDs
    resolved = resolved_df[~resolved_df["symbol"].str.startswith(("Unresolved", "LOC"))]
    unresolved = resolved_df[resolved_df["symbol"] == "Unresolved"]
    loc_ids = resolved_df[resolved_df["symbol"].str.startswith("LOC")]

    # Save resolved IDs to CSV
    resolved.to_csv(resolved_file, index=False)
    print(f"Resolved gene IDs saved to {resolved_file}")

    # Save unresolved IDs to CSV
    unresolved.to_csv(unresolved_file, index=False)
    print(f"Unresolved gene IDs saved to {unresolved_file}")

    # Save "LOC" IDs to CSV
    loc_ids.to_csv(loc_file, index=False)
    print(f"'LOC' gene IDs saved to {loc_file}")

def query_ncbi_loc(loc_id: str) -> dict:
    """
    Query NCBI E-utilities API to retrieve metadata for LOC IDs.
    Args:
        loc_id (str): LOC ID to query.
    Returns:
        dict: Dictionary containing metadata for the LOC ID.
    """
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        search_url = f"{base_url}/esearch.fcgi?db=gene&term={loc_id}&retmode=json"
        response = requests.get(search_url)
        if response.ok:
            search_data = response.json()
            idlist = search_data.get("esearchresult", {}).get("idlist", [])
            if idlist:  # Check if idlist is not empty
                gene_id_ncbi = idlist[0]
                summary_url = f"{base_url}/esummary.fcgi?db=gene&id={gene_id_ncbi}&retmode=json"
                summary_response = requests.get(summary_url)
                if summary_response.ok:
                    summary_data = summary_response.json()
                    if "result" in summary_data and gene_id_ncbi in summary_data["result"]:
                        gene_data = summary_data["result"][gene_id_ncbi]
                        return {
                            "loc_id": loc_id,
                            "name": gene_data.get("name", "Unknown"),
                            "description": gene_data.get("description", "No description available"),
                            "other_designations": gene_data.get("otherdesignations", "None")
                        }
        return {"loc_id": loc_id, "name": "Unresolved", "description": "Unresolved", "other_designations": "Unresolved"}
    except Exception as e:
        print(f"NCBI API error for {loc_id}: {e}")
        return {"loc_id": loc_id, "name": "Error", "description": "Error", "other_designations": "Error"}

def identify_loc_ids(input_file: str, output_file: str):
    """
    Identify LOC IDs using NCBI E-utilities API and save results to a file.
    Args:
        input_file (str): Path to the input file containing LOC IDs.
        output_file (str): Path to save identified LOC IDs with metadata.
    """
    try:
        loc_df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
        return

    loc_ids = loc_df["symbol"].tolist()
    identified_data = []

    print(f"Identifying {len(loc_ids)} LOC IDs using NCBI E-utilities API...")
    for loc_id in loc_ids:
        metadata = query_ncbi_loc(loc_id)
        identified_data.append(metadata)

    # Save identified LOC IDs with metadata to a CSV file
    identified_df = pd.DataFrame(identified_data)
    identified_df.to_csv(output_file, index=False)
    print(f"Identified LOC IDs saved to {output_file}")

if __name__ == "__main__":
    input_file = "unresolved_gene_ids.txt"
    resolved_file = "resolved_gene_ids2.csv"
    unresolved_file = "unresolved_gene_ids.csv"
    loc_file = "loc_gene_ids.csv"
    resolve_gene_ids(input_file, resolved_file, unresolved_file, loc_file)

    input_file = "loc_gene_ids.csv"
    output_file = "identified_loc_gene_ids.csv"
    identify_loc_ids(input_file, output_file)