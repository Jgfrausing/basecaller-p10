# Improving Oxford Nanopore Technologies (ONT) MinION base calling S9 project

## The JKBC library
The `jkbc` library contains various helper functions related to the project.
It is a package, which makes it easy to install. If you plan to edit the package while using it,
use the following command to install it (while being in the *outer* `jkbc` folder):
`pip install -e .`

## CLAAUDIA

### Connecting
Generally, `<username>` is your AAU email. Commands supposed to be run in a terminal is preceeded with `$`. 

#### Inside the AAU network OR using VPN/Gateway:

```
$ ssh <username>@ai-pilot.srv.aau.dk
```

#### Outside the AAU network:
The easiest way is to add the following to your ssh config file, usually located at `$HOME/.ssh/config`:
```
Host claaudia
 HostName ai-pilot.srv.aau.dk
 User <username>
 ProxyJump <username>@sshgw.aau.dk
```

Afterwards, you can simply use the following command to connect (`claaudia` is the name given on the `Host` line in the ssh config):
```
$ ssh claaudia
```

The quick and greedy approach without setup requires two steps.

First you connect to the "gateway server".
```
$ ssh <username>@sshgw.aau.dk
```
Then, once inside the gateway, you can use the regular ssh command as if you were inside AAU.

### Setting Up the Container
Pull the PyTorch container (this will take a while):
```
$ srun singularity pull docker://nvcr.io/nvidia/pytorch:20.02-py3
```

[See a list of available containers here](https://ngc.nvidia.com/catalog/containers?orderBy=&query=&quickFilter=deep-learning&filters=).

### Running and Accessing the Container with Jupyter Lab
This method requires that you are inside the AAU network (VPN is possible) to access Jupyter Lab.

In the **CLAAUDIA frontend terminal**:
  1. Check the `node-id` of the node you are currently using, which can be seen by simply looking at your prompt, as it should have the name `<username>@nv-ai-fe<node-id>:~$`, most likely this is `01` or `03`.
  2. Run Jupyter Lab in the PyTorch container:
    ```
    $ srun --pty --gres=gpu:1 singularity exec -B $HOME/runuser/:/run/user/$(id -u) --nv pytorch_20.02-py3.sif jupyter lab --port=8860 --ip=0.0.0.0
    ```
  3. The previous command should output a `token` which must be used when accessing the Jupyter Lab. 
      * It is usually of the form `http://localhost:8860/?token=<token>`
      * Copy this token (which is a string of ASCII characters)

On a computer **in the AAU network/on the VPN**:
  * Access Jupyter Lab on the website `http://nv-ai-<node-id>.srv.aau.dk:8860/lab?token=<token>` with the `node-id` and `token` replaced.


## Various tips
* Save your conda environment to a file: `conda env export | grep -v "^prefix: " > <NAME_OF_FILE>.yml`
* Create a new conda environment using a file: `conda env create -f <NAME_OF_FILE>.yml`

**Following section is untested and is likely to change:**
* Setting up a continually running Jupyter Lab
  * You should be inside the CLAAUDIA Frontend server
  * Open/start a tmux session (this will make sure it never closes):
    * Check if a 'jupyter' session already exists: `$ tmux ls`
    * If not, create one: `$ tmux new -s jupyter` 
    * Else, open it: `$ tmux a -t jupyter`
    * Regardless of whether you created a new tmux or simply opened another one, you should now be in a special tmux terminal. 
  * Everything that follows will be in the tmux terminal:
    * Open a shell in the container: `$ srun --gres=gpu:1 --pty singularity shell --nv pytorch_20.02-py3.sif`
    * Load the bashrc file so conda works: `$ source /user/student.aau.dk/<user>/.bashrc`
      * On first run you need to let conda alter the rc file first: `conda MISSING COMMAND`
    * You should now see the conda `(base)`-prefix in your terminal.
    * Run the Jupyter Lab: `jupyter lab --port=8860 --ip=0.0.0.0`
    * Copy the token that is printed, and open a the Lab as described in the *accessing jupyter lab section* above.
    * Detach (exit with killing) the tmux terminal by pressing `ctrl+b` and then pressing `d`

  
**Adding a conda environment as a kernel in Jupyter Lab:**
  * Assuming the conda environment you want to add is called `jkbc`, use the following commands:
```
conda activate jkbc
(jkbc)$ ipython kernel install --user --name=<any_name_for_kernel>
```

## Walkthrough: Initial run
1. Connect to the AAU VPN. 
2. SSH into the CLAAUDIA frontend: `ssh <username>@ai-pilot.srv.aau.dk`
3. Create the Singularity image (like a docker image): `srun singularity pull docker://nvcr.io/nvidia/pytorch:20.02-py3`
4. Create the runuser directory (usage unknown): `mkdir runuser`
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


## Walkthrough: Subsequent runs
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
8. Remember to choose the correct *kernel* for your notebooks (jkbc)
