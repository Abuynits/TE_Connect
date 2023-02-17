"""
Plan:
get a data collumn to compare and always compare against the price prediction
have the option, similar to the other one to visually split the data into subplots nad see how weel trend follows
"""
from data_constants import OUTPUT_DATA_COLS

OUTPUT_DATA_COL = OUTPUT_DATA_COLS[0]
# filter based on what you want to compare and view
DATA_FILTER = ["Asia Pacific & ANZ", "Channel - Industrial"]
# the external indicator used
EXTERNAL_FACTOR = "sales_quantity"
# determine whether to analyze plots in batches or individually
PLOT_INDIVIDUALLY = False
ROWS_DISPLAY = 2
COLS_DISPLAY = 2
FIGURE_WIDTH = 10
FIGURE_HEIGHT = 6
