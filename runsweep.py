import os
import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-g", "--gpus", default=1)
  parser.add_argument("-s", "--sweep_cmd")
  args = parser.parse_args()

  for _ in range(int(args.gpus)):
    command = "cd ~/basecaller-p10/nbs/basecaller && conda activate jkbc && %s &" % args.sweep_cmd
    os.system("srun --gres=gpu:1 singularity exec --nv jkbc.simg %s" % command)

if __name__ == "__main__":
  main()
