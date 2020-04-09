# Improving Oxford Nanopore Technologies (ONT) MinION base calling S9 project

## The JKBC library
The `jkbc` library contains various helper functions related to the project.
It is a package, which makes it easy to install. If you plan to edit the package while using it,
use the following command to install it (while being in the *outer* `jkbc` folder):
`pip install -e .`

---

## CLAAUDIA
Note that placeholders in this guide is shown as a name inside angle brackets (e.g. `<placeholder>`), and they are, along with the brackets, to be replaced by actual values. The placeholder `<username>` is most likely your AAU student email (e.g. xxxx@student.aau.dk).

### Connecting
Before you can connect to the CLAAUDIA server, you need to be on the AAU VPN–*this is necessary even if you are on the AAU network*. [Follow this guide on how to setup and use the AAU VPN](https://www.its.aau.dk/vejledninger/vpn/)

Once you are on the VPN you can ssh into the CLAAUDIA server using the following command in a terminal.

```
$ ssh <username>@ai-pilot.srv.aau.dk
```

### Walk-through: Initial run
1. Connect to the AAU VPN. 
2. SSH into the CLAAUDIA frontend: `ssh <username>@ai-pilot.srv.aau.dk`
3. Create the Singularity image (like a docker image): `srun singularity pull docker://nvcr.io/nvidia/pytorch:20.02-py3`
   * [See a list of alternative containers here](https://ngc.nvidia.com/catalog/containers?orderBy=&query=&quickFilter=deep-learning&filters=).
4. Create the runuser directory (*usage yet unknown*): `mkdir runuser`
5. Open a tmux terminal named "jupyter", which allows you to exit without terminating jobs: `tmux new -s jupyter`
6. Run a shell in the container: `srun --gres=gpu:1 --pty singularity shell -B $HOME/runuser/:/run/user/$(id -u) --nv pytorch_20.02-py3.sif`
7. Clone git repo: `git clone https://github.com/Jgfrausing/basecaller-p10.git` 
8. Create a conda environment from the conda_env.yml file in the repo: 
    * Cd into the repository folder: `cd basecaller-p10`
    * Create a new conda environment using the conda_env.yml file: `conda env create -f conda_env.yml` (this will take a while)
9. Create a Jupyter kernel that uses the new conda environment.
    * Setup conda for bash: `conda init bash`
    * Load the created .bashrc file: `source /user/student.aau.dk/jfraus14/.bashrc` (replace username)
    * Activate the conda environment: `conda activate jkbc`
    * Create the kernel for jupyter `ipython kernel install --user --name=jkbc`
10. Install the JKBC library locally using pip: `pip install -e jkbc/`
11. Run Jupyter Lab `jupyter lab --port=8860 --ip=0.0.0.0`
    * It will print out the port actually used (might differ if occupied) and an access token.
    * For example `http://127.0.0.1:<port>/?token=<token>`
    * Figure out the id of the node you are running on by looking in your prompt.
      * Its name will be something like `kbargs15@student.aau.dk@nv-ai-fe<id>:~$`.
    * Make save these three pieces of information: id, port, and token
12. Detach (exit without kill) from the tmux terminal: press `ctrl+b` and then `d`
13. Connect to the Jupyter Lab in your browser
    * While being on the AAU network, connect to: `http://nv-ai-<id>.srv.aau.dk:<port>/lab?token=<token>`
14. Choose the correct kernel for your notebooks, i.e. the `jkbc` kernel that we created earlier.


### Walk-through: Subsequent runs
1. Connect to the AAU VPN.
2. SSH into the CLAAUDIA frontend: `ssh <username>@ai-pilot.srv.aau.dk`
3. Open a tmux terminal by either creating a new session or attaching to an
   existing one:
    * Create a new session called "jupyter": `tmux new -s jupyter`
    * Attach a terminal to an existing session called "jupyter": `tmux attach -t
     jupyter`
4. Run a shell in the container: `srun --gres=gpu:1 --pty singularity shell -B $HOME/runuser/:/run/user/$(id -u) --nv pytorch_20.02-py3.sif`
5. Run Jupyter Lab: `jupyter lab --port=8860 --ip=0.0.0.0`
    * See step *11.* in the *Initial run* walthrough for how to find id, port
    and token.
6. Detach (exit without kill) from the tmux terminal: press `ctrl+b` and then `d`
7. Connect to the Jupyter Lab in your browser with values from step 5.:
   `http://nv-ai-<id>.srv.aau.dk:<port>/lab?token=<token>`
8. Remember to choose the correct *kernel* for your notebooks (jkbc).

### Keep jupyter running
It is important, that you keep the tab open in order to be able to continue execution
1. Detach from the tmux running jupyter: `Ctrl + b` + `d`
2. Disconnect from the aau connection: `exit`
3. Turn of VPN
#### Reconnecting
1. Log into VPN and ssh into aau
2. Reconnect kernel by `Kernel` -> `Reconnect kernel`
---

## Conda Environment Files
* Save your conda environment to a file: `conda env export | grep -v "^prefix: " > <NAME_OF_FILE>.yml`
* Create a new conda environment using a file: `conda env create -f <NAME_OF_FILE>.yml`

## Make new basecaller
A script is created to ensure the correct structure of the notebook. This makes experimentation with models easier.

The following practices should be adhered to:
* Run `./new-basecaller <name-of-basecaller>` to create a new basecaller
    * This will create a folder in *models-nbk* with the given name, and a notebook within that folder.
* The field under *Build* should be the only place that differs between notebooks. 
    * This makes it easier to adjust for changes made in pre- or post-processing, if needed.
* The models are saved and loaded using the name of the notebook.
    * Rename the notebook whenever changes are made and a new model is trained
        * Run `git co -- <name-of-old-notebook>.ipynb`
        * and `git add <name-of-new-notebook>.ipynb`
        * this ensures that both versions of the model are saved and their traind weights can be loaded

## Make and load training data
We created a script that reads hdf5 files, and saves training data into feather files. This greatly improves the speed that data is loaded into memory, and skips a lot of the preprocessing. 
### Make
1. Activate conda environment: `conda activate jkbc`
2. Run `python [path to make_feather_file.py] [path to data] [OPTIONS]`
    * By default the script will create a small example and save it to data/feather-files/
    * The following options can be applied:
        - `--f [number]` and `--t [number]`: The range of signals to be used (default 0 and 5)
        - `--ll`: The fixed size of the labels (default 70)
        - `--o`: Override the path to save the files
        - `-run_test`: Runs a test that ensures that saved data equals loaded data
    * To avoid defining where the save the files, run the script from the root of this project
        - `python jkbc/jkbc/utils/make_feather_file.py [path to data] [OPTIONS]`
3. The script creates a folder named `Range0-50-FixLabelLen70`
    * Where:
        - 0 corresponds to the `--f` parameter
        - 50 corresponds to the `--t` parameter
        - 70 corresponds to the `--ll` parameter
    * Within the folder lies the files:
        - x (input data)
        - y (labels)
