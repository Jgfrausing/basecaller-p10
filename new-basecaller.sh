#!/bin/bash
if [ "$#" -eq  "1" ]
	then
		nbPath="model-nbk/$1/$1.ipynb"
		if test -f "$nbPath"; 
		then
	 		echo "A basecaller with that name already exist"
			exit 1
		fi
		mkdir -p model-nbk/$1
		cp model-nbk/template/Model-template.ipynb $nbPath 
		echo $1 created
	else
		echo Missing parameter: name
fi
