# load in the data and mount it on the drive
from src.filepaths_constants import *
from data_processing import *
from visualization import *
from saving_reading_data import *

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
display_group_df(grouped_df)

reg_data, transformed_data, input_transformations, output_transformations, check_transforms_key = transform_norm_rem_out(
    grouped_df, INPUT_DATA_COLS, OUTPUT_DATA_COLS, DATA_FILTER)

# display a sample of original data, transformed data, and un transformed data
if CHECK_DATA_TRANSFORMS:
    # display(transformed_data[k])
    check_data_transformations(check_transforms_key, reg_data, transformed_data, output_transformations)

save_all_transformations(input_transformations, output_transformations)
save_all_data(reg_data, transformed_data)
print("saved transformations and data!")