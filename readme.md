# Code for the Paper "How to Query Language Models?"

## Setup conda environment
```
conda create --name lmkb python=3.8
conda activate lmkb
pip install -r requirements.txt
```

## Download the data
```
# LAMA
wget https://dl.fbaipublicfiles.com/LAMA/data.zip -O data/temp.zip
unzip data/temp.zip -d data/
mv data/data data/LAMA
rm data/temp.zip

# BATS
wget https://vecto-data.s3-us-west-1.amazonaws.com/BATS_3.0.zip -O data/temp.zip
unzip data/temp.zip -d data/
rm data/temp.zip
```

## Do a single run over one of LAMA's corpora
```
CUDA_VISIBLE_DEVICES=0 python main.py \
--model bert-large-cased \
--data TREx \
--num_priming_examples 10 \
--priming_type nl \
--use_close_examples \
--nosave_results
```
If you happen to have more GPUs available and want to speed up inference, you can just specify multiple devices, e.g.,
```
CUDA_VISIBLE_DEVICES=0,1,2 pythom main.py # ...
```

## Run Experiments
The experiment script has the default behavior of parallelizing over the available GPUs with individual random seeds while sequentially looping through all other hyperparameter options (as specified in experiments.sh). To achieve this it creates a tmux session with n windows where n="number of GPUs". The results are saved as pandas DataFrames in the specified output directory.
```
bash run_experiments.sh 'out/results'
```
To summarize the data of the individual runs in a single DataFrame, we use:
```
python data/merge_data.py --results_dir out/results --output_file out/results/summary.csv
```

From this summary, you can now generate the table and plots of the paper as follows:
```
python results/results.py --summary_file out/results/summary.csv --output_dir out/results/summary/
```
