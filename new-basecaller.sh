#!/bin/bash
mkdir model-nbk/$1
cp model-nbk/template/Model-template.ipynb model-nbk/$1/$1.ipynb
echo $1 created
