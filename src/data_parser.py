import pandas as pd
from eval import *
import re

df = pd.read_csv("/Users/abuynits/PycharmProjects/TE_Connect/data/results.csv")

all_pred = df["pred"].dropna()
all_pred = all_pred.str.replace('tensor', '').replace('[()\[\]]', '', regex=True).str.replace(' ', '')

all_actual = df["actual"].dropna()
all_actual = all_actual.str.replace('tensor', '').str.replace('[()\[\]]', '', regex=True).str.replace(' ', '')

all_actual = all_actual.to_numpy()
all_pred = all_pred.to_numpy()

assert (len(all_actual) == len(all_pred))

product_codes = df.iloc[2::3, :]['year'].to_numpy()
print(len(product_codes))
print(product_codes[0])
business_groups = df.iloc[::3, :]['business group'].to_numpy()
print(len(business_groups))
print(business_groups[0])
regions = df.iloc[::3, :]['region'].to_numpy()
print(len(regions))
print(regions[0])
years = df.iloc[0::3, :]['year'].str.replace('[\[\]]', '', regex=True).to_numpy()
print(len(years))
print(years[0])
months = df.iloc[0::3, :]['month'].str.replace("\n", "", regex=False) \
    .str.replace("[ ", "", regex=False) \
    .str.replace("[", "", regex=False) \
    .str.replace("]", "", regex=False) \
    .str.replace("  ", " ").to_numpy()
print(len(months))
print(months[0])

master_pred = []
master_actual = []
month_master_pred = []
month_master_actual = []
all_data = []
all_data.append("product code,business group,region,year,month,actual,prediction")
for l in range(len(all_actual)):
    actual = list(map(float, all_actual[l].split(",")))
    pred = list(map(float, all_pred[l].split(",")))
    year = list(map(float, years[l].split(" ")))
    print(len(months[l]))
    month = list(map(float, months[l].split(" ")))

    if len(actual) != len(pred):
        continue
    for i in range(len(pred)):
        if (i + 1) % 4 == 0:
            acc_avg = sum(actual[i - 3:i]) / 4
            pred_avg = sum(pred[i - 3:i]) / 4
            print(product_codes[l])
            print(type(product_codes[l]))
            p_code = re.sub("['()]", "", product_codes[l])
            p_code = p_code.split(",")[0]
            all_data.append(

                f"{p_code},{business_groups[l]},{regions[l]},{year[i]},{month[i]},{acc_avg},{pred_avg}")
            month_master_pred.append(pred_avg)
            month_master_actual.append(acc_avg)
    master_actual.extend(actual)
    master_pred.extend(pred)

print(len(master_actual))
print(len(master_pred))
print(len(all_data))

print(calc_feature_similarity(month_master_actual, month_master_pred))
print(calc_feature_similarity(master_actual, master_pred))

df_write = pd.DataFrame(all_data)
df_write.to_csv('results2.csv', sep=";", index=False)
