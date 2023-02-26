import pickle

from factor_analysis_constants import *
from data_processing import *
from visualization import *

assert len(OUTPUT_DATA_COLS) == 1, "OUTPUT data cols should only be 1 value!"
if READ_FROM_DRIVE:
    from google.colab import drive

    drive.mount('/content/gdrive')
    df = pd.read_csv(CSV_DRIVE_PATH)
else:
    df = pd.read_csv(CSV_FILE_PATH)

print("num rows:", len(df))
print("num cols:", df.shape[1])
# adds in another column that is the time data for the dataframe
df = alter_df_time_scale(df)
display(df.head(10))

# extract the filtered data based on the business unit gropu, comany region and product line
raw_df = get_raw_data(df)
display(raw_df.head(10))

# get a grouped data frame on each one of the critical factors
grouped_df = group_by_unique(df,
                             product_line=GROUP_BY_PRODUCT_LINE,
                             business_unit=GROUP_BY_BUSINESS_UNIT,
                             company_region=GROUP_BY_COMPANY_REGION)

# display some of the groups in the grouped dataframe
# display_group_df(grouped_df)

# Generate all of the data that is used - only care about transformed data and graphing the
"""
Generate all of the data that is used
only care about the transformed input cols and output cols
"""
_, transformed_data, _, _, _ = transform_norm_rem_out(
    grouped_df, [OUTPUT_DATA_COL], [OUTPUT_DATA_COL], DATA_FILTER)
# only care about transformed data as it contains all values need for comparison
assert len(transformed_data) != 0, "no elements detected for filter!"
print(len(transformed_data))

if CMP_ALL_EXTERNAL_DF:
    print(f"comparing all external variables at {EXTERNAL_DF_FP}")
    external_df = pd.read_csv(EXTERNAL_DF_FP)
    all_external_indicators = external_df.columns.values
    all_transformed_external_data = transform_norm_rem_out(external_df, all_external_indicators)
    factor_to_product, product_to_factor = get_all_factor_comparison(transformed_data,
                                                                     [OUTPUT_DATA_COL],
                                                                     all_transformed_external_data)
    print(factor_to_product.keys())
    print(product_to_factor.keys())
    with open(FACTOR_TO_PRODUCT_DICT_FP, "w") as outfile:
        json.dump(factor_to_product, outfile)
    with open(PRODUCT_TO_FACTOR_DICT_FP, "w") as outfile:
        json.dump(product_to_factor, outfile)
else:
    if CMP_EXTERNAL_DF:
        print(f"comparing external variable: {EXTERNAL_DF_VAR} from df at {EXTERNAL_DF_FP}!")
        external_df = pd.read_csv(EXTERNAL_DF_FP)  # only has a single variable
        transformed_external_data = transform_norm_rem_out(external_df, [EXTERNAL_DF_VAR])

    # visual comparison display:
    if PLOT_INDIVIDUALLY:
        for key, value in transformed_data.items():
            if not CMP_EXTERNAL_DF:
                display_factor_comparison(key, value, OUTPUT_DATA_COL, EXTERNAL_FACTOR)
            else:
                raise Exception("comparison with other frames for individual plots not supported!")
    else:
        if not CMP_EXTERNAL_DF:
            display_multiple_factors_comparison(transformed_data,
                                                [OUTPUT_DATA_COL],
                                                [EXTERNAL_FACTOR],
                                                ROWS_DISPLAY,
                                                COLS_DISPLAY,
                                                FIGURE_WIDTH,
                                                FIGURE_HEIGHT)
        else:
            display_multiple_factors_comparison(transformed_data,
                                                [OUTPUT_DATA_COL],
                                                [EXTERNAL_DF_VAR],
                                                ROWS_DISPLAY,
                                                COLS_DISPLAY,
                                                FIGURE_WIDTH,
                                                FIGURE_HEIGHT,
                                                transformed_external_data)
