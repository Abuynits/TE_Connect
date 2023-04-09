import zipfile
with zipfile.ZipFile("te_ai_data", 'r') as zip_ref:
    zip_ref.extractall("te_ai_data")