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