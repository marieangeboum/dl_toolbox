#!/bin/bash

echo "#############################################################"
echo `python --version`
echo "#############################################################"
echo "Params to train.sh : "
echo "$@"
echo "#############################################################"


python ~/SemiSupervised/SSL4Remote/dl_toolbox/main.py "$@"