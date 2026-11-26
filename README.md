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
Keep in mind that cold start takes a couple of seconds, after this inference performance for this query yields:  
query: 'Coindexter'
results: 5  (138.5 ms total; 7.6 ms lexical; 130.7 ms semantic)
candidates: 41 lexical / 100 semantic

## REPL
Because we're currently using a network drive the actual startup time for semantic search is a bit long. Every time we load the model from disk we have to ship 2.3GB through the network into RAM. This will be solved once we deploy since we will save the onnx data to NVME and keep the model loaded in RAM for repeated queries. 
We have yet to implement the actual API for search on demand, but we can get a feeling for actual performance via 
python -m semanticsearch.repl --threads 16