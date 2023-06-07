import requests
import redis
import os
r = redis.Redis(host='localhost', port=6379, db=0)

WEBHOOK_URL = r.get('ngrok_url').decode() + '/'
ASSEMBLYAI_API_TOKEN = "API_KEY_GOES_HERE"

if ASSEMBLYAI_API_TOKEN == "API_KEY_GOES_HERE":
    print('Please set your API key in transcribe.py')
    exit()


def read_file(filename, chunk_size=5242880):
    print('Reading file: '+ filename)
    with open(filename, 'rb') as _file:
        while True:
            data = _file.read(chunk_size)
            if not data:
                break
            yield data


def upload_file(filename):
    print('Uploading file: {}'.format(filename))
    headers = {'authorization': ASSEMBLYAI_API_TOKEN}
    response = requests.post('https://api.assemblyai.com/v2/upload',
                            headers=headers,
                            data=read_file(filename))
    # print(response.json())
    url = response.json()['upload_url']
    # print('Upload URL: {}'.format(url))
    return url

def create_transcript(url, filename):
    # print('Creating transcript for: {}'.format(url))
    endpoint = "https://api.assemblyai.com/v2/transcript?filename="+filename
    json = {
        "audio_url": url,
        "punctuate": False,
        "format_text": False,
        "webhook_url": WEBHOOK_URL,
    }
    headers = {
        "authorization": ASSEMBLYAI_API_TOKEN,
    }
    response = requests.post(endpoint, json=json, headers=headers)
    r.rpush('job_order', response.json().get('id'))
    return response.json()

def get_transcript(id):
    # print('Getting transcript for: {}'.format(id))
    endpoint = "https://api.assemblyai.com/v2/transcript/{}".format(id)
    headers = {'authorization': ASSEMBLYAI_API_TOKEN}
    response = requests.get(endpoint, headers=headers)
    return response.json()
    

    