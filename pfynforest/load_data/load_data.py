import pandas as pd

def load_data(file_path="pfynforest/data/raw/envidatpfyncrowncond2016.xlsx"):

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