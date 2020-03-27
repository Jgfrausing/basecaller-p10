#!/bin/bash
if [ "$#" -eq  "1" ]
	then
		mkdir model-nbk/$1
		cp model-nbk/template/Model-template.ipynb model-nbk/$1/$1.ipynb
		echo $1 created
	else
		echo Missing parameter: name

fi
