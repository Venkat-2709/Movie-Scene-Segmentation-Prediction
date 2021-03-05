# Eluv.io_ML_Challenge

## The model will the predict the scene segmentation. 

### To execute the code

Download the repo and copy the data file into the project directory and name the folder as 'data'.  

### Project Structure

```bash
    |--data
    |    |--all the data file (.pkl)
    |--main.py
    |--evaluate.py
    |--results
        |--all the results (.pkl)
    
```

Then for executing the model type the following command in terminal:

```
python main.py
```

Once the code finished executing, the model is trained and the predicted results are stored in results folder as 'imdb_id.pkl' files.
###### Note: This might take few minutes to finish executing (20-30 mins) Since the data very large to process.

Already trained and predicted results are stored in results folder. To evaluate the model run the following command:

```
python evaluate.py results
```

This will give the evaluation results for model. 

This is output of ``` python evaluate.py results``` : 
```
# of IMDB IDs: 64
Scores: {
    "AP": 0.9635864130938265,
    "mAP": 0.9630648280427021,
    "Miou": 0.8844696713127517,
    "Precision": 0.9452091353014112,
    "Recall": 0.8573620747774839,
    "F1": 0.8980189641004256
}
```
These scores may vary a little bit. 
