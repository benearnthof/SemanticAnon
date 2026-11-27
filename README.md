# SemanticAnon
Semantic Search for the TrueAnon podcast

## Project Setup
Execute these commands for initial setup.
```bash
git clone https://github.com/benearnthof/SemanticAnon.git  
cd SemanticAnon
chmod +x setup.sh
./setup.sh
source /workspace/SemanticAnon/activate.sh
```
To download & build the dataset from patreon you'll need a subscription. yt_dlp can scrape off patreon, but to do so on a remote VM the respective cookies.txt from your browser of choice is required.
```bash
yt-dlp --cookies-from-browser firefox "URL"
```
Check out the yt_dlp docs for more.


## Episode Ingest
Move the cookies file to the correct location first so we can ingest episodes directly.
```bash
mv /workspace/cookies.txt /workspace/SemanticAnon/cookies.txt
```
Also provide secrets.env with the huggingface token necessary for diarization.  
```https://huggingface.co/pyannote/speaker-diarization-community-1```
Check out the pyannote Huggingface space for more information, but in a nutshell you need a file `secrets.env` that contains a single line with your Huggingface token for the community model download like so:  
```bash
HF_TOKEN="YOUR_PYANNOTE_TOKEN"
```
After this you can begin episode ingest, which downloads, transcribes, and diarizes all podcast episodes. You can set parameters like maximum speakers and batch size for whisper via the cli, or just use default values like so:  
```bash
python ./scripts/episode_ingest.py --batch-size 64
```
This, of course, requires GPU inference. Diarization is the most expensive part, transcription runs at around 200x real time. In our experience diarization was throughput limited and required only modest amounts of GPU memory, which is why parallelization on cheaper GPUs is prefered over renting a single B300.  
This downloads, transcribes, & diarizes new episodes. 

## Annotating New Episodes
After diarization has been done, which might take a couple of hours for the entire dataset, one can manually add labels to identify speakers. We did this in a small active learning loop by first labeling 50 randomly sampled episodes via a streamlit interface, then training a logistic regression classifier to identify the two hosts. Prelabeling the hosts from this data already saves around 50% of total annotation effort, since they make up the vast majority of spoken time. After prelabeling the two hosts, an additional 50 episodes were manually annotated, sampled, this time, by the total "non-confident pre-annotation time" meaning the more content contained in these episodes that could not be confidentyl identified by the logistic regression classifier, the higher the priority for manual labeling. The goal of this was to invest manual lableing effort precisely in episodes with potential high information gain. This seemed to work well as proceeding in this manner drastically cut down total labeling time and led to the last 50% of the dataset lacking only 1-2 labels or being already confidently labeled completely. As a funny aside, the most "difficult" episode to label with the highest potential information gain identified in this way was the April fools episode machine-translated to Chinese.  
After 350-400 episodes were manually labeled we switched over to a residual autoencoder for pre-labeling, which further decreased manual effort by correctly identifying speaker classes like "YOUNG CHOMSKY", "CALLER" (for hate-line episodes), "MIX" (for overlapping speakers), and "SOUNDBITE" (for music jingles and bits).  

To update database indices we do not need to prelabel new episodes, since usually a handful of episodes can be labeled in a couple of minutes via this lightweight (and vibe coded) streamlit app.
```bash
./labeling/app_ingest.py.
```
If you wish to use predictive models for pre-labeling you may do so by first training a classifier and then assigning pre-annotations like so:  
```bash
python scripts/diarization_models/train_classifier.py
python scripts/diarization_models/predict_speakers.py
```
The lightweight ingest app does not, however, use predictions. You'll need to use an older, less performant, version of the streamlit interface.  
You can now proceed with updating the lexical search database & embeddings needed for hybrid search.  

## Updating Lexical & Semantic Search Indices  
Because we use hybrid search for best results we have to both update the lexical search index and the semantic embedding matrix. To embed documents we used Qwen-0.6B, which seems to yield good results for such a small model and initial results.
```bash
python ./semanticsearch/update_index.py
```
We will experiment with additional embedding models, as well as various re-ranking methods later.  
TODO: Re-ranking
TODO: Late-Interaction Models

## Export Embedding Model to ONNX for Fast Inference
We want to keep everything as light-weight as possible. In the ideal case a small, CPU-only embedding api would embed user queries. We experimented with a couple of inference strategies, so far onnx with 4-8 threads yields the best results on CPU.  
We first have to export our model to the onnx format, before being able to embed user queries.
```bash
python ./semanticsearch/export_onnx.py
```
Make sure you are using the correct onnx runtime:
```bash
# GPU inference
uv add sentence-transformers[onnx-gpu]
# CPU inference
uv add sentence-transformers[onnx]
# In case of a version mismatch you can run
uv sync --reinstall-package onnxruntime --all-extras
```

## Running a search query
After exporting your embedding model of choice to onnx you can now begin running queries against the dataset. 
```bash
python -m semanticsearch.hybrid.search "Coindexter" -n 5 --episodes --context 2 --support-limit 3
```
Keep in mind that running searches in this way will load the model from disk, query the model, and then discard the model from RAM after search results are returned.  
To keep the model in RAM and only incur cold-start time once use the REPL or API.  
TODO: implement API  
In any case, results should look like this on a modern CPU:  
```
query: 'Coindexter'
results: 5  (138.5 ms total; 7.6 ms lexical; 130.7 ms semantic)
candidates: 41 lexical / 100 semantic
```

## REPL
Because we're currently using a network drive the actual startup time for semantic search is a bit long. Every time we load the model from disk we have to send 2.3GB through the network into RAM. This will be solved once we deploy since we will save the onnx data to NVME and keep the model loaded in RAM for repeated queries. 
We have yet to implement the actual API for search on demand, but we can get a feeling for actual performance with the minimal repl interface:  
```bash
python -m semanticsearch.repl --threads 16
```

## Search API
```bash
uv run uvicorn semanticsearch.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```
