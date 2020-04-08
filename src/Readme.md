## Shahar Lutati and Omer Ber
## Tel Aviv University
# Introduction 

We have 2 files:
Lenet5.py which consisit the different modules ('none','dropout','batch normalization','weight decay')
where 'weight decay' is actually the 'none' with wd !=0.

RunFunc.py which is the main function. This function runs with CLI meaning it parse arguments from
the cmd. We worked a lot so the handeling will be as smooth as possible.

for example if we open a cmd and type
## python RunFunc.py -h## 
the output:

usage: RunFunc.py [-h] [-t {all,none,dropout,bn,wd}] [-b BATCH_SIZE] [-e EPOCHS] [-p DATA_PATH] [-v] [-l]

optional arguments:
  -h, --help            show this help message and exit
  -t {all,none,dropout,bn,wd}, --technique {all,none,dropout,bn,wd}
                        Technique used for training, default <all>
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size, default <256>
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs, default <15>
  -p DATA_PATH, --data_path DATA_PATH
                        Path to database, default <"./data/mnist">
  -v, --verbose         Enable verbose mode
  -l, --test_mode       Load models for testing only

so for example if we want to train a model, for example 'bn' which stands for batch normalization with batch size of 256, and 15 epochs
"python RunFunc.py -t bn -b 256 -e 15"
please note that there are default values as well.

## Special notes
The RunFunc.py first checkes if in the data_path (if not entered ,there's a default value) exists the data,
if not it downloads it from mnist/fashion and export it to the data_path folder (also create it in case it does'nt exists)

After training the function will save the best model according to the test results (before overfitting).
The training models are saved to './models/model_name'.

Two important flags :
-v verbose execution - will draw a progress bar during the run.
-l test mode - wil access to './models/ and look for the specific name given by '-t' flag to test on.