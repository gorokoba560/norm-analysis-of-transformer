#!/bin/sh

echo "Running data_extract.py ..."
python ./data_extract.py

echo "Running check_dispersion.py ..."
python ./check_dispersion.py

echo "Running re-examine.py ..."
python ./re-examine.py

echo "Running relation_between_alpha_fx.py ..."
python ./relation_between_alpha_fx.py

echo "Running relation_with_freq.py ..."
python ./relation_with_freq.py

echo "Done!"