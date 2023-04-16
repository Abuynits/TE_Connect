import pandas as pd
import glob

path = r'/Users/abuynits/PycharmProjects/TE_Connect/data/datasets_top_5'
allFiles = glob.glob(path + "/*.csv")
allFiles.sort()
df_out = pd.DataFrame()

for file in allFiles:
    df = pd.read_csv(file, header=None, skiprows=1)
    print(file)
    df_out = df_out.append(df, ignore_index=True)
# hopefully, this will drop blank rows
df_out.to_csv("/Users/abuynits/PycharmProjects/TE_Connect/data/data.csv", index=False)
# write to file
