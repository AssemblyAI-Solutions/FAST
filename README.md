# ASR but make it Fast 🏃

### How to get started

1. Install dependencies
```python
pip install -r requirements.txt
```
2. Start Redis Server on CLI (Current config uses localhost)
```shell
redis-server
```

3. Run ```output.py```, this is responsible for printing transcripts in order
```
python3 output.py
```

4. Run ```app.py```, this is responsible for listening for AssemblyAI Webhooks, sent on transcript completion.
```
python3 app.py
```

5. Run ```chunked_asr.py```, this is responsible for pulling Audio from URL and chunking into smaller files using Voice Activity Detection (VAD)

```
python3 chunked_asr.py
```