#!/bin/bash

cd .cache
wget -O Tensor.tar.gz https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:tensor_factorisation_datasets.tar.gz
tar -xf Tensor.tar.gz
rm Tensor.tar.gz