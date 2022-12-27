#!/bin/bash

if [ ! -e base1b.fbin ]; then
    echo "Dwonloading base data"
    azcopy copy https://comp21storage.blob.core.windows.net/publiccontainer/comp21/MSFT-TURING-ANNS/base1b.fbin .
fi

if [ ! -e query100K.fbin ]; then
    echo "Dwonloading query data"
    azcopy copy https://comp21storage.blob.core.windows.net/publiccontainer/comp21/MSFT-TURING-ANNS/query100K.fbin .
fi

if [ ! -e query_gt100.bin ]; then
    echo "Dwonloading ground truth"
    azcopy copy https://comp21storage.blob.core.windows.net/publiccontainer/comp21/MSFT-TURING-ANNS/query_gt100.bin .
fi
