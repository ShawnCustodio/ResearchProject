import scanpy as sc
import pandas as pd
import numpy as np
import requests
import time
from mygene import MyGeneInfo
from typing import Dict, List
import xml.etree.ElementTree as ET

def clean_gene_id(gene_id):
    """Strip version number from gene IDs"""
    if '.' in gene_id:
        return gene_id.split('.')[0]
    return gene_id

def get_gene_names_ensemble(gene_ids: List[str], batch_size: int = 50) -> Dict[str, str]:
    """Query multiple APIs in sequence for gene information"""
    all_symbols = {}
    remaining_ids = gene_ids.copy()
    
    # Step 1: Try MyGeneInfo REST API
    print("Step 1: Querying MyGeneInfo REST API...")
    mygene_symbols = get_mygene_names(remaining_ids, batch_size)
    all_symbols.update(mygene_symbols)
    
    # Get remaining unresolved IDs
    remaining_ids = [g for g in remaining_ids if mygene_symbols.get(g, g) == g]
    if remaining_ids:
        # Step 2: Try Ensembl REST API
        print(f"\nStep 2: Querying Ensembl REST API for {len(remaining_ids)} remaining IDs...")
        ensembl_symbols = get_ensembl_names(remaining_ids, batch_size)
        all_symbols.update(ensembl_symbols)
        
        # Get remaining unresolved IDs
        remaining_ids = [g for g in remaining_ids if ensembl_symbols.get(g, g) == g]
        if remaining_ids:
            # Step 3: Try GENCODE REST API
            print(f"\nStep 3: Querying GENCODE for {len(remaining_ids)} remaining IDs...")
            gencode_symbols = get_gencode_names(remaining_ids)
            all_symbols.update(gencode_symbols)
            
            # Get remaining unresolved IDs
            remaining_ids = [g for g in remaining_ids if gencode_symbols.get(g, g) == g]
            if remaining_ids:
                # Step 4: Try HGNC REST API
                print(f"\nStep 4: Querying HGNC for {len(remaining_ids)} remaining IDs...")
                hgnc_symbols = get_hgnc_names(remaining_ids)
                all_symbols.update(hgnc_symbols)

                remaining_ids = [g for g in remaining_ids if hgnc_symbols.get(g, g) == g]
                if remaining_ids:
                    # Step 5: Try NCBI E-utilities API
                    print(f"\nStep 5: Querying NCBI E-utilities API for {len(remaining_ids)} remaining IDs...")
                    ncbi_symbols = get_ncbi_names(remaining_ids)
                    all_symbols.update(ncbi_symbols)

                    remaining_ids = [g for g in remaining_ids if ncbi_symbols.get(g, g) == g]
                    if remaining_ids:
                        # Step 6: Try UniProt API
                        print(f"\nStep 6: Querying UniProt API for {len(remaining_ids)} remaining IDs...")
                        uniprot_symbols = get_uniprot_names(remaining_ids)
                        all_symbols.update(uniprot_symbols)
    
    return all_symbols

def get_mygene_names(gene_ids: List[str], batch_size: int = 1000) -> Dict[str, str]:
    """Query MyGene.info for gene names"""
    mg = MyGeneInfo()
    all_symbols = {}
    not_found = []
    
    print(f"Processing {len(gene_ids)} gene IDs in batches of {batch_size}...")
    for i in range(0, len(gene_ids), batch_size):
        batch = gene_ids[i:i + batch_size]
        clean_batch = [clean_gene_id(g) for g in batch]
        
        try:
            results = mg.querymany(clean_batch, 
                                 scopes=['ensembl.gene', 'symbol', 'alias'],
                                 fields=['symbol', 'name', 'type_of_gene', 'gene_biotype'],
                                 species='human',
                                 as_dataframe=True)
            
            batch_dict = {}
            for original_id, clean_id in zip(batch, clean_batch):
                # Handle duplicate hits by checking if clean_id is in results.index
                matching_rows = results[results.index == clean_id]
                if len(matching_rows) > 0:
                    row = matching_rows.iloc[0]
                    symbol = row['symbol'] if 'symbol' in row and pd.notna(row['symbol']) else ''
                    name = row['name'] if 'name' in row and pd.notna(row['name']) else ''
                    biotype = row['gene_biotype'] if 'gene_biotype' in row and pd.notna(row['gene_biotype']) else ''
                    
                    if symbol and isinstance(symbol, str):
                        batch_dict[original_id] = symbol
                    elif name and isinstance(name, str):
                        if biotype == 'lncRNA':
                            batch_dict[original_id] = f"{name} (lncRNA)"
                        else:
                            batch_dict[original_id] = name
                    else:
                        batch_dict[original_id] = original_id
                        not_found.append(original_id)
                else:
                    batch_dict[original_id] = original_id
                    not_found.append(original_id)
                    
            all_symbols.update(batch_dict)
            print(f"Processed batch {i}-{min(i+batch_size, len(gene_ids))}")
            
        except Exception as e:
            print(f"Warning: Error in batch {i}-{i+batch_size}: {str(e)}")
            all_symbols.update(dict(zip(batch, batch)))
    
    if not_found:
        print(f"\nGenes not found in MyGeneInfo ({len(not_found)}):")
        print(', '.join(not_found[:10]) + ('...' if len(not_found) > 10 else ''))
            
    return all_symbols

def get_ensembl_names(gene_ids: List[str], batch_size: int = 50) -> Dict[str, str]:
    """Original Ensembl REST API query"""
    base_url = "https://rest.ensembl.org"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Python-Research-Project"
    }
    all_symbols = {}
    not_found = []
    
    print(f"Processing {len(gene_ids)} gene IDs in batches of {batch_size}...")
    for i in range(0, len(gene_ids), batch_size):
        batch = gene_ids[i:i + batch_size]
        batch_dict = {}
        
        for gene_id in batch:
            clean_id = clean_gene_id(gene_id)
            try:
                url = f"{base_url}/lookup/id/{clean_id}?expand=1"
                response = requests.get(url, headers=headers)
                
                if response.ok:
                    data = response.json()
                    display_name = data.get('display_name', '')
                    biotype = data.get('biotype', '')
                    
                    if display_name:
                        if biotype == 'lncRNA':
                            batch_dict[gene_id] = f"{display_name} (lncRNA)"
                        else:
                            batch_dict[gene_id] = display_name
                    else:
                        batch_dict[gene_id] = gene_id
                        not_found.append(gene_id)
                else:
                    batch_dict[gene_id] = gene_id
                    not_found.append(gene_id)
                
                # Respect API rate limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Warning: Error processing {gene_id}: {str(e)}")
                batch_dict[gene_id] = gene_id
                not_found.append(gene_id)
        
        all_symbols.update(batch_dict)
        print(f"Processed batch {i}-{min(i+batch_size, len(gene_ids))}")
    
    if not_found:
        print(f"\nGenes not found in Ensembl ({len(not_found)}):")
        print(', '.join(not_found[:10]) + ('...' if len(not_found) > 10 else ''))
    
    return all_symbols

def get_gencode_names(gene_ids: List[str]) -> Dict[str, str]:
    """Query GENCODE/HAVANA database"""
    base_url = "https://www.gencodegenes.org/human/api"
    symbols = {}
    
    for gene_id in gene_ids:
        clean_id = clean_gene_id(gene_id)
        try:
            url = f"{base_url}/gene/{clean_id}"
            response = requests.get(url)
            
            if response.ok:
                data = response.json()
                symbol = data.get('gene_symbol', '')
                biotype = data.get('gene_type', '')
                
                if symbol:
                    if biotype == 'lncRNA':
                        symbols[gene_id] = f"{symbol} (lncRNA)"
                    else:
                        symbols[gene_id] = symbol
                else:
                    symbols[gene_id] = gene_id
            else:
                symbols[gene_id] = gene_id
                
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"GENCODE API error for {gene_id}: {str(e)}")
            symbols[gene_id] = gene_id
    
    return symbols

def get_hgnc_names(gene_ids: List[str]) -> Dict[str, str]:
    """Query HGNC database"""
    base_url = "https://rest.genenames.org"
    symbols = {}
    
    for gene_id in gene_ids:
        clean_id = clean_gene_id(gene_id)
        try:
            url = f"{base_url}/fetch/ensembl_gene_id/{clean_id}"
            headers = {'Accept': 'application/json'}
            response = requests.get(url, headers=headers)
            
            if response.ok:
                data = response.json()
                if data['response']['numFound'] > 0:
                    doc = data['response']['docs'][0]
                    symbol = doc.get('symbol', '')
                    if symbol:
                        symbols[gene_id] = symbol
                    else:
                        symbols[gene_id] = gene_id
                else:
                    symbols[gene_id] = gene_id
            else:
                symbols[gene_id] = gene_id
                
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"HGNC API error for {gene_id}: {str(e)}")
            symbols[gene_id] = gene_id
    
    return symbols

def get_ncbi_names(gene_ids: List[str]) -> Dict[str, str]:
    """Query NCBI E-utilities API"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    symbols = {}
    
    for gene_id in gene_ids:
        clean_id = clean_gene_id(gene_id)
        try:
            url = f"{base_url}/esearch.fcgi?db=gene&term={clean_id}&retmax=1&usehistory=y"
            response = requests.get(url)
            
            if response.ok:
                root = ET.fromstring(response.content)
                ids = [elem.text for elem in root.findall(".//Id")]
                
                if ids:
                    gene_id_ncbi = ids[0]
                    url_summary = f"{base_url}/esummary.fcgi?db=gene&id={gene_id_ncbi}&retmode=xml"
                    response_summary = requests.get(url_summary)
                    
                    if response_summary.ok:
                        root_summary = ET.fromstring(response_summary.content)
                        name = root_summary.find(".//Name").text
                        symbols[gene_id] = name if name else gene_id
                    else:
                        symbols[gene_id] = gene_id
                else:
                    symbols[gene_id] = gene_id
            else:
                symbols[gene_id] = gene_id
                
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"NCBI API error for {gene_id}: {str(e)}")
            symbols[gene_id] = gene_id
    
    return symbols

def get_uniprot_names(gene_ids: List[str]) -> Dict[str, str]:
    """Query UniProt API"""
    base_url = "https://rest.uniprot.org/uniprotkb/stream"
    symbols = {}
    
    for gene_id in gene_ids:
        clean_id = clean_gene_id(gene_id)
        try:
            params = {
                "query": f"gene:{clean_id}",
                "format": "tsv",
                "fields": "gene_names"
            }
            response = requests.get(base_url, params=params)
            
            if response.ok:
                lines = response.text.splitlines()
                if len(lines) > 1:
                    gene_names = lines[1].split('; ')
                    if gene_names:
                        symbols[gene_id] = gene_names[0]  # Take the first gene name
                    else:
                        symbols[gene_id] = gene_id
                else:
                    symbols[gene_id] = gene_id
            else:
                symbols[gene_id] = gene_id
                
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"UniProt API error for {gene_id}: {str(e)}")
            symbols[gene_id] = gene_id
    
    return symbols

def preprocess_data(input_file, output_file, n_samples=1000, min_cells=10, target_sum=10000):
    """
    Preprocesses single-cell RNA-seq data from an AnnData object.
    """
    print("Loading data...")
    adata = sc.read_h5ad(input_file, backed='r')
    adata_subset = adata[:n_samples].to_memory()
    
    print("Filtering and normalizing...")
    sc.pp.filter_genes(adata_subset, min_cells=min_cells)
    sc.pp.normalize_total(adata_subset, target_sum=target_sum)
    sc.pp.log1p(adata_subset)

    df = adata_subset.to_df()
    
    print("Converting gene IDs to symbols...")
    if 'feature_name' in adata_subset.var.columns:
        # First pass: use feature names where available
        name_dict = {}
        unresolved_ids = []
        for idx, row in adata_subset.var.iterrows():
            if pd.notna(row['feature_name']):
                name_dict[idx] = row['feature_name']
            else:
                unresolved_ids.append(idx)
                name_dict[idx] = idx  # temporary placeholder
        
        # Second pass: use ensemble of APIs for remaining Ensembl IDs
        if unresolved_ids:
            print(f"Looking up {len(unresolved_ids)} Ensembl IDs using multiple APIs...")
            gene_symbols = get_gene_names_ensemble(unresolved_ids)
            name_dict.update(gene_symbols)
        
        gene_names = [name_dict[g] for g in df.columns]
        
        # Identify remaining unresolved IDs
        remaining_unresolved_ids = [gene_id for gene_id, name in name_dict.items() if name.startswith(('ENSG', 'ENST', 'NM_', 'NR_'))]
    else:
        # Use ensemble of APIs for all IDs
        gene_symbols = get_gene_names_ensemble(df.columns.tolist())
        gene_names = [gene_symbols.get(g, g) for g in df.columns]
        
        # Identify remaining unresolved IDs
        remaining_unresolved_ids = [gene_id for gene_id, name in gene_symbols.items() if name.startswith(('ENSG', 'ENST', 'NM_', 'NR_'))]
    
    # Update column names
    df.columns = gene_names
    
    # Output unresolved IDs to a file
    unresolved_file = "unresolved_gene_ids.txt"
    with open(unresolved_file, "w") as f:
        for gene_id in remaining_unresolved_ids:
            f.write(f"{gene_id}\n")
    print(f"Unresolved gene IDs output to {unresolved_file}")
    
    # Save processed data
    print(f"Saving processed data to {output_file}...")
    df.to_csv(output_file)
    
    # Print summary of unresolved IDs
    print(f"\nSummary: {len(remaining_unresolved_ids)} gene IDs could not be resolved.")
    print("Done!")

if __name__ == "__main__":
    input_file = "datav2.h5ad"
    output_file = "pp_data_ID6.csv"
    preprocess_data(input_file, output_file)



