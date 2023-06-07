import ngrok
import redis

ngrok_tunnel = ngrok.start_tunnel(5000)
print('Public URL:', ngrok_tunnel)
r = redis.Redis(host='localhost', port=6379, db=0)
r.set('ngrok_url', ngrok_tunnel)



from flask import Flask, request
import logging
import metrics
import transcribe


first_transcript_flag = True

# create Flask app
app = Flask(__name__)

# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)


@app.route('/', methods=['POST'])
def webhook_handler():
    global first_transcript_flag
    if first_transcript_flag:
        metrics.measure_first_transcript_latency()
        first_transcript_flag = False
    try:
        job_id = request.json.get('transcript_id')
        print('job_id: ' + job_id)
        test_id = r.get('test_id').decode()
        last_job = r.get('test_id:' + str(test_id) + ':last_job').decode()
        print('last_job: ' + last_job)
        print('job_id: ' + job_id)
        if last_job == job_id:
            metrics.measure_complete_and_ordered()
    except:
        pass
    transcript = transcribe.get_transcript(job_id)
    payload = transcript.get('text')
    if job_id and payload:
        r.hset('job_results', job_id, payload)  # store the payload in a hash
    return {'message': 'Webhook received'}, 200



if __name__ == "__main__":

    # start the Flask app
    app.run(port=5000)
