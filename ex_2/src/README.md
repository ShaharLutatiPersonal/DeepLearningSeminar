## Shahar Lutati and Omer Ber
## Tel Aviv University
# Introduction 

## This Drive is Read-Only, you should copy all the files to your local machine

1. Upload the data (train,val,test) to folder in the local machine under './ptb/data'
2. Upload the traind models under './ptb/models' (optional if no training is desired)
3. Run the sections above train_and_test to compile it.
4. Run train_and_test with the following arguments:
chosen_models - 'all'/'lstm'/'gru'
epochs - integer 
verbose bool
is_test_mode bool - if you want to run train model from the models uploaded in (2)
data_path - path to data folder ('./data')

The function will run for dropout of 0,0.35.
If you want to save the models + zip file of the graphs you can run the last section.

some example of output generated from the models:
"but many expect hong kong 's issue in the exchange was fractionally by lower prices for the fiscal first consecutive week of N their session"
