## TRISECT (Time-seRIeS forECasting Toolkit)

### What is TRISECT

TRISECT is a simple and automated way to execute various deep learning models for time-series forecasting including:
- LSTM
- GRU
- BiDirectional LSTM
- CNN LSTM

---

#### Configuration File

In `config.json` file, several variables can be configured:

	"epochs": ,
	"batch_size": ,
	"test_size": ,
	"scaler": ,
	"timestamp_column":,
	"X":"",
	"Y": "",
	"num_past_samples": "",
	"num_future_samples": ""

`epochs`: an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation

`batch_size`: the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need

`train_size`: this parameter decides the size of the data that has to be split as the train dataset

`scaler`: the scaler that will be used for the data. Either "MinMax" or "Standard" scaler can be used

`timestamp_column`: The csv files may sometimes contain columns related to the timestamp of each sample. Indicate the name of that column to remove it

`X`: Indicate the name of the columns that will be used as features. One or more columns can be used

`Y`: Indicate the name of the columns that the DL models need to forecast. One or more columns can be used

`num_past_samples`: Indicate how many time steps in the past the DL models need to take into account when forecasting

`num_future_samples`: Indicate how many time steps in the future the DL models need to forecast

---

###### _An example cmd_
```shell
python forecast.py -d daily-minimum-temperatures-in-me.csv -c config.json
```

After the execution of the desired model, the following plots are created and saved:

- Training/Test loss

##### Python version used

- Python 3.9.7 Interpreter