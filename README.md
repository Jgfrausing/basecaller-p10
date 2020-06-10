## Install JKBC
1. Cd into the repository folder: `cd basecaller-p10`
2. Create a conda environment from the conda_env.yml file in the repo: 
    * Create a new conda environment using the conda_env.yml file: `conda env create -f conda_env.yml` (this will take a while)
3. Activate the conda environment: `conda activate jkbc`
4. Install the JKBC library locally using pip: `$CONDA_PREFIX/bin/pip install -e jkbc`


## Making predictions
1. Activate the conda environment if not active: `conda activate jkbc`
2. cd into basecaller: `cd nbs/basecaller`
3. Run the prediction script `python predict.py <id> <data_set> <name_of_run>`
  * Any id from https://app.wandb.ai/jkbc/jk-basecalling-v2 can be used, however the models presented in the report are:
    * JKBC-1: 2eiadj4y
    * JKBC-2: 1ywu3vo9
    * JKBC-3: 2d84exku
    * JKBC-4: j6f2sn3v
    * JKBC-5: 1c2vr2my
  * A small test set is include in nbs/basecaller/test-data/
  * To predict using JKBC-5 use the command: `python predict.py 1c2vr2my test-data`
  * This creates the folder 1c2vr2my-test-data/ containing reference.fasta and predictions.fasta.
