from factor_analysis_constants import *
from filepaths_constants import *
from saving_reading_data import *
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
    grouped_df, [OUTPUT_DATA_COL], [EXTERNAL_FACTOR], DATA_FILTER)
# only care about transformed data as it contains all values need for comparison
assert len(transformed_data) != 0, "no elements detected for filter!"
# visual comparison display:
if PLOT_INDIVIDUALLY:
    for key, value in transformed_data.items():
        display_factor_comparison(key, value, OUTPUT_DATA_COL, EXTERNAL_FACTOR)
else:
    display_multiple_factors_comparison(transformed_data,
                                    [OUTPUT_DATA_COL],
                                    [EXTERNAL_FACTOR],
                                    ROWS_DISPLAY,
                                    COLS_DISPLAY,
                                    FIGURE_WIDTH,
                                    FIGURE_HEIGHT)

