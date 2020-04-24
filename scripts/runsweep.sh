#!/bin/bash

for ((i = 0; i < $1; i++))
do
	CUDA_VISIBLE_DEVICES=$1 cd ~/basecaller-p10/nbs/basecaller; $2 &
done

