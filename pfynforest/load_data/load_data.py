import pandas as pd

def load_data(file_path="pfynforest/data/raw/envidatpfyncrowncond2016.xlsx"):
    """
    Load and preprocess forest data from an Excel file containing multiple sheets.
    
    This function reads tree data from an Excel file with multiple sheets, each representing
    a different subset of the data. It extracts relevant columns related to tree characteristics
    and crown defoliation measurements, concatenates data from all sheets, and removes rows
    with missing defoliation data.
    
    Parameters:
        file_path (str, optional): Path to the Excel file containing the forest data.
                                  Defaults to "pfynforest/data/raw/envidatpfyncrowncond2016.xlsx".
    
    Returns:
        pandas.DataFrame: A preprocessed DataFrame containing the following columns:
            - TREE NO: Unique tree identifier
            - PLOT NO: Plot number where the tree is located
            - TREATMENT: Treatment type applied to the tree or plot
            - X: X-coordinate of the tree
            - Y: Y-coordinate of the tree
            - TOTAL CROWN DEFOLIATION: Percentage of crown defoliation
            - SOCIAL STATUS: Social status of the tree in the forest hierarchy
    
    Notes:
        - Skips the 'HEADER' sheet if present
        - Only includes sheets with the required columns
        - Filters out rows with missing defoliation data
    """
    # Get list of all sheets in the Excel file
    xl = pd.ExcelFile(file_path)
    print("Available sheets:", xl.sheet_names)

    all_data = {}
    for sheet in xl.sheet_names:
        if sheet == 'HEADER':
            continue
        else:
            df = pd.read_excel(file_path, sheet_name=sheet)
            if 'TREE HEIGHT MEASURED' not in df.columns:
                all_data[sheet] = df[['TREE NO', 'PLOT NO','TREATMENT', 'X', 'Y','TOTAL CROWN DEFOLIATION','SOCIAL STATUS']]
    df = pd.concat(all_data.values())
    df = df.dropna(subset=["TOTAL CROWN DEFOLIATION"])
    return df