# Predicting Customer Churn
We say that a customer has "churned" if they have stopped buying from us.


This code trains an SVM or LSTM neural network on customer purchase data to predict customer churn. 
The LSTM includes embedding layers to learn semantic representations of non-numerical features. 

You may define a search space for hyper-parameter tuning.

## Prerequisites
cd into the project directory and install the project requirements in ./requirements.txt.

## Creating feature matrices 
Next, move your customer order file into the ./data directory. 
Edit the default parameters in ./data_writer.py's class initializer.
Finally, run 
```
$ python ./data_writer.py 
```
The code creates two 3-dimensional X matrices. 
The first x_flt.npy holds the numeric features. The second x_str.npy holds the non-numeric features.

They have shape #customers x customer.#orders x #numeric_features.

The Y matrix has a 1 if the customer churned and a 0 otherwise. 
It has dimension #customers x 1.


## Running Models
### SVM
Run 
```
$ cd ./models
```

and in svm.py, set the search space for hyper-parameter tuning. 
Use the lists in the search_dict at the bottom of the file.
Run 
```
$ python svm.py
```

### LSTM
cd into the models directory. 
In lstm.py, define the search space for hyper-parameter tuning: 
use the lists in the search_params dictionary at the bottom of the file.
Run 
```
$ python lstm.py
```

To see the tensorboard plots, run 

```
$ python -m tensorboard.main --logdir=churn_dir_path/models/logs/lstm/file/
```

## 
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
