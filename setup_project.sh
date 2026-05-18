#!/bin/bash
mkdir -p src/ingestion src/mining src/parameterization src/evaluation notebooks experiments config
touch src/__init__.py src/ingestion/__init__.py src/mining/__init__.py src/parameterization/__init__.py src/evaluation/__init__.py
cat <<EOF > requirements.txt
pandas
pyarrow
numpy
scikit-learn
openai
tqdm
plotly
nbformat
EOF
