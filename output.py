import time
import redis
import metrics
r = redis.Redis(host='localhost', port=6379, db=0)


def process_jobs():
    first_flag = True
    while True:
        job_id = r.lindex('job_order', 0)  # get the first job id from the list
        if job_id:
            if first_flag:
                # metrics.measure_first_transcript_latency()
                first_flag = False
            job_id = job_id.decode()  # Redis returns bytes, so we need to decode to str
            payload = r.hget('job_results', job_id)  # get the payload from the hash
            if payload:
                payload = payload.decode()  # Redis returns bytes, so we need to decode to str
                print(f"{payload}", flush=True)
                r.lpop('job_order')  # remove the job id from the list
                r.hdel('job_results', job_id)  # remove the payload from the hash
        time.sleep(.1)  # sleep to prevent the loop from running too fast

process_jobs()