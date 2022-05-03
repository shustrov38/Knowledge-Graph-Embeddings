#!/bin/bash

cd .cache
wget -O wordnet-mlj12.tar.gz https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz 
tar -xf wordnet-mlj12.tar.gz
rm wordnet-mlj12.tar.gz