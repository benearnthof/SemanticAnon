# SemanticAnon
Semantic Search for the TrueAnon podcast

## Project Setup
git clone https://github.com/benearnthof/SemanticAnon.git
cd SemanticAnon
chmod +x setup.sh
./setup.sh
source /workspace/SemanticAnon/activate.sh

## Episode Ingest
Make sure to generate cookies.txt with yt_dlp first so download works.  
mv /workspace/cookies.txt /workspace/SemanticAnon/cookies.txt
Also provide secrets.env with the huggingface token necessary for diarization.
python ./scripts/episode_ingest.py --batch-size 1
This downloads, transcribes, & diarizes new episodes.

## Annotating New Episodes
To update indices we need fully labeled data. Resolving speaker identities is done via the app in ./labeling/app_ingest.py. 
We can also train predictive models to help us with labeling.  
python scripts/diarization_models/train_classifier.py
python scripts/diarization_models/predict_speakers.py
(TODO: add absolute paths to these such that the scripts work out of the box).
After prelabeling we assign labels via the streamlit interface. We can then proceed with updating the lexical search database & embeddings needed for hybrid search.
streamlit run ./labeling/app_ingest.py

## Updating Lexical & Semantic Search Indices
python ./semanticsearch/update_index.py

## Export Embedding Model to ONNX for Fast Inference
python ./semanticsearch/export_onnx.py
make sure to add the correct onnx runtime
uv add sentence-transformers[onnx-gpu]
or 
uv add sentence-transformers[onnx] for later CPU inference.
run this in case of version mismatch
uv sync --reinstall-package onnxruntime --all-extras

## Running a search query
python -m semanticsearch.hybrid.search "Coindexter" -n 5 --episodes --context 2 --support-limit 3

