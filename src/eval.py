from visualization import *
from saving_reading_data import*
dict_train_data,dict_valid_data,dict_test_data = read_dicts_from_fp()
input_transformations, output_transformations = read_transformations_from_fp()
model = read_model_from_fp()
reg_data, transformed_data = read_data_from_fp()

for key, val in enumerate(dict_test_data):
  print (val)
print()
for key, val in enumerate(dict_train_data):
  print (val)
print()
for key, val in enumerate(dict_valid_data):
  print (val)

show_all_model_prediction(dict_test_data,transformed_data,output_transformations,model,PREDICT_DISPLAY_COUNT)