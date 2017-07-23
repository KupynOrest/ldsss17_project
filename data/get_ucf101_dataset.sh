#!/bin/bash

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo ${CURRENT_DIR}
cd ${CURRENT_DIR}

FILE=UCF101.rar
DIR=ucf101
URL=http://crcv.ucf.edu/data/UCF101/UCF101.rar

if [ -f ${FILE} ] || [ -d ${DIR} ]; then
    echo "file exists"
else 
    echo "* downloading dataset ..."
    wget ${URL} -O ${FILE}
fi

