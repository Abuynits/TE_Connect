from data_constants import *


def get_raw_data(filtered_df, business_unit_group_name=None, company_region_name_level1=None, product_line_code=None):
    if business_unit_group_name is not None:
        filtered_df = filtered_df[filtered_df['business_unit_group_name']
                                  == business_unit_group_name]
    if company_region_name_level1 is not None:
        filtered_df = filtered_df[filtered_df['company_region_name_level_1']
                                  == company_region_name_level1]
    if product_line_code is not None:
        filtered_df = filtered_df[filtered_df['product_line_code']
                                  == product_line_code]

    if len(filtered_df) == 0:
        raise Exception("ERROR: empty dataset")
    return filtered_df


def alter_df_time_scale(df):
    df["year_week_ordered"] = df['fiscal_year_historical'] * 100 + \
                              df['fiscal_week_historical']
    df.sort_values(by=['fiscal_year_historical'] +
                      ['fiscal_week_historical'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    u_prod_code = df["product_line_code"].unique()
    print(u_prod_code)
    df["Price"] = df["sales_amount"] / df["sales_quantity"]
    return df


def group_by_unique(local_df, product_line=False, business_unit=False, company_region=False):
    group_by_list = []
    if (product_line):
        group_by_list.append("product_line_code")
    if (company_region):
        group_by_list.append("company_region_name_level_1")
    if (business_unit):
        group_by_list.append("business_unit_group_name")
    grouped_df = local_df.groupby(group_by_list)
    print(len(grouped_df))
    return grouped_df


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"

    df.dropna(inplace=True)
    print("new len:", len(df.columns))
    return df


# TODO: need to implement removal of outliers
def removeOutliers(group, col):
    return group


def key_filter_match(name, data_filter):
    if len(data_filter) == 0:
        return True
    for e in data_filter:
        if not e in name:
            return False
    return True


def transform_norm_rem_out(grouped_df, input_data_cols, output_data_cols, data_filter):
    transformed_data = {}
    reg_data = {}
    input_transformations = {}
    output_transformations = {}
    display_once = True
    count = 0
    data_cols = input_data_cols + list(set(output_data_cols) - set(input_data_cols))
    print(input_data_cols)
    print(output_data_cols)

    save_first_name = False
    check_transforms_key = ""

    for name, group in grouped_df:
        if save_first_name:
            check_transforms_key = name
            save_first_name = False

        if key_filter_match(name, data_filter):
            for col in data_cols:

                if len(group[col]) == 0:
                    raise Exception("Collumn does not exist!")
                # removeOutliers(group[col],col)
                # interpolate the mission values

            scalar1 = MinMaxScaler()
            scalar2 = MinMaxScaler()
            # group[data_cols] = clean_dataset(group[data_cols]) # this line
            for i in range(0, len(group.columns)):
                group.iloc[:, i].interpolate(inplace=True)
            group = group.replace((np.inf, -np.inf, np.nan),
                                  0).reset_index(drop=True)

            reg_data[name] = group.copy()
            input_transformations[name] = scalar1.fit(group[input_data_cols])
            output_transformations[name] = scalar2.fit(group[output_data_cols])
            group[data_cols] = scalar1.fit_transform(group[input_data_cols])
            transformed_data[name] = group
            if display_once:
                print("\n")
                print(name)
                display(transformed_data[name])
                display_once = False
            count += 1
            print(count, "/", len(grouped_df), end="\r")

    return reg_data, transformed_data, input_transformations, output_transformations, check_transforms_key


def prep_data_for_transformer_model(data, past, future, input_data_cols, output_data_cols):
    x, target, y = list(), list(), list()
    inp_data_arr = data[input_data_cols].to_numpy()
    out_data_arr = data[output_data_cols].to_numpy()
    for i in range(min(len(inp_data_arr), len(out_data_arr))):
        lag_end = i + past
        forcast_end = lag_end + future
        target_start = lag_end - 1
        target_end = forcast_end - 1
        if forcast_end > len(data):
            break
        x.append(inp_data_arr[i:lag_end])
        target.append(inp_data_arr[target_start:target_end])
        y.append(out_data_arr[lag_end:forcast_end])
    return np.array(x), np.array(target), np.array(y)


def prep_data_for_model(data, past, future, input_data_cols, output_data_cols):
    x, y = list(), list()
    inp_data_arr = data[input_data_cols].to_numpy()
    out_data_arr = data[output_data_cols].to_numpy()
    for i in range(min(len(inp_data_arr), len(out_data_arr))):
        lag_end = i + past
        forcast_end = lag_end + future
        if forcast_end > len(data):
            break
        x.append(inp_data_arr[i:lag_end])
        y.append(out_data_arr[lag_end:forcast_end])
    return np.array(x), np.array(y)


def split_data(transformed_data, dict_train_data, dict_valid_data,
               dict_test_data, all_valid_data, all_train_data,
               all_test_data):
    show_sample = SHOW_SAMPLE_TEST_TRAIN_VAL_SPLIT
    for key, group in transformed_data.items():
        train_test_split = random.random()
        valid_data_split = random.random()
        if train_test_split < TEST_TRAIN_SPLIT:

            if valid_data_split < TEST_TRAIN_SPLIT:
                # train_x: all input sequence
                # train_target: consists of last data point of the input to the one before the last of target seq
                # train_y: the desired output of the sequence for computing loss
                train_x, train_target, train_y = prep_data_for_transformer_model(group,
                                                                                 LOOKBACK,
                                                                                 PREDICT,
                                                                                 INPUT_DATA_COLS,
                                                                                 OUTPUT_DATA_COLS)
                all_train_data.append((train_x, train_target, train_y))
                dict_train_data[key] = (train_x, train_target, train_y)
                if show_sample:
                    print("train:")
                    print(train_x.shape)
                    print(train_x[0])
                    print(train_target.shape)
                    print(train_target[0])
                    print(train_y.shape)
                    print(train_y[0])
                    # print(y_train[0])
                    # show_sample = False
            else:
                x_valid, target_valid, y_valid = prep_data_for_transformer_model(group,
                                                                                 LOOKBACK,
                                                                                 PREDICT,
                                                                                 INPUT_DATA_COLS,
                                                                                 OUTPUT_DATA_COLS)
                all_valid_data.append((x_valid, target_valid, y_valid))
                dict_valid_data[key] = (x_valid, target_valid, y_valid)
                if show_sample:
                    print("valid:")
                    print(x_valid.shape)
                    print(x_valid[0])
                    print(target_valid.shape)
                    print(target_valid[0])
                    print(y_valid.shape)
                    print(y_valid[0])
                    # print(y_valid[0])
                    show_sample = False
        else:
            x_test, target_test, y_test = prep_data_for_transformer_model(group,
                                                                          LOOKBACK,
                                                                          PREDICT,
                                                                          INPUT_DATA_COLS,
                                                                          OUTPUT_DATA_COLS)
            all_test_data.append((x_test, target_test, y_test))
            dict_test_data[key] = (x_test, target_test, y_test)
            if show_sample:
                print("test:")
                print(x_test.shape)
                print(x_test[0])
                print(target_test.shape)
                print(target_test[0])
                print(y_test.shape)
                print(y_test[0])


def get_all_data_arr(all_data):
    tr = []
    target = []
    te = []
    for i in range(len(all_data)):
        x_tr = all_data[i][0]
        x_tg = all_data[i][1]
        x_te = all_data[i][2]
        for j in range(len(x_tr)):
            tr.append(x_tr[j])
            target.append(x_tg[j])
            te.append(x_te[j])
    return np.array(tr), np.array(target),np.array(te)
