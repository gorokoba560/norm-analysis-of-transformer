#!/bin/sh

if [ ! -e ./data/europarl-v7.de-en.de ]; then
  wget -nc -P ./data http://statmt.org/europarl/v7/de-en.tgz
  tar -xvzf ./data/de-en.tgz -C ./data
  rm ./data/de-en.tgz
fi

if [ ! -e ./data/DeEnGoldAlignment.tar.gz ]; then
  echo "Download gold alignment data from: https://www-i6.informatik.rwth-aachen.de/goldAlignment/index.php#download"
elif [ ! -d ./data/DeEn ]; then
  tar -xvzf ./data/DeEnGoldAlignment.tar.gz -C ./data
fi