from eval import *


SEQ2SEQ = "/Users/abuynits/PycharmProjects/TE_Connect/data/results_seq2seq.csv"
DEEP_ESN = "/Users/abuynits/PycharmProjects/TE_Connect/data/results_deepecn.csv"
model_choice = SEQ2SEQ
write_csv = True
OUTPUT_FP = "/Users/abuynits/PycharmProjects/TE_Connect/data/parser_out.csv"


df = pd.read_csv(model_choice)

all_pred = df["pred"].dropna()
all_pred = all_pred.str.replace('tensor', '').replace('[()\[\]]', '', regex=True).str.replace(' ', '')

all_actual = df["actual"].dropna()
all_actual = all_actual.str.replace('tensor', '').str.replace('[()\[\]]', '', regex=True).str.replace(' ', '')

all_actual = all_actual.to_numpy()
all_pred = all_pred.to_numpy()

assert (len(all_actual) == len(all_pred))

product_codes = df.iloc[2::3, :]['year'].to_numpy()
business_groups = df.iloc[::3, :]['business group'].to_numpy()
regions = df.iloc[::3, :]['region'].to_numpy()
years = df.iloc[0::3, :]['year'].str.replace('[\[\]]', '', regex=True).to_numpy()
months = df.iloc[0::3, :]['month'].str.replace("\n", "", regex=False) \
    .str.replace("[ ", "", regex=False) \
    .str.replace("[", "", regex=False) \
    .str.replace("]", "", regex=False) \
    .str.replace("  ", " ").to_numpy()

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
    month = list(map(float, months[l].split(" ")))

    if len(actual) != len(pred):
        continue
    for i in range(len(pred)):
        if (i + 1) % 4 == 0:
            acc_avg = sum(actual[i - 3:i]) / 4
            pred_avg = sum(pred[i - 3:i]) / 4
            p_code = re.sub("['()]", "", product_codes[l])
            p_code = p_code.split(",")[0]
            all_data.append(

                f"{p_code},{business_groups[l]},{regions[l]},{year[i]},{month[i]},{acc_avg},{pred_avg}")
            month_master_pred.append(pred_avg)
            month_master_actual.append(acc_avg)
    master_actual.extend(actual)
    master_pred.extend(pred)

month_acc, month_bias, _ = calc_feature_similarity(month_master_actual, month_master_pred)
day_acc, day_bias, _ = calc_feature_similarity(master_actual, master_pred)

if model_choice == SEQ2SEQ:
    print("========= SEQ2SEQ =========")
elif model_choice == DEEP_ESN:
    print("========= DEEP_ESN =========")

print("month acc:{:.4f}, month bias: {:.4f}".format(month_acc, month_bias))
print("day acc:{:.4f}, day bias: {:.4f}".format(day_acc, day_bias))

if write_csv:
    df_write = pd.DataFrame(all_data)
    df_write.to_csv(OUTPUT_FP, sep=";", index=False)
