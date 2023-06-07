import redis
import time
r = redis.Redis(host='localhost', port=6379, db=0)

def start_new_test():
    # Create new id for test in Redis and return
    test_id = r.incr('test_id')
    r.set('test_id:' + str(test_id) + ':start_time', time.time())
    return test_id


def measure_first_transcript_latency():
    # Get test id from Redis
    test_id = r.get('test_id').decode()
    # Set first transcript time from Redis
    r.set('test_id:' + str(test_id) + ':first_transcript_time', time.time())


def measure_complete_and_ordered():
    # Get test id from Redis
    test_id = r.get('test_id').decode()
    # Set end time from Redis
    end_time = r.set('test_id:' + str(test_id) + ':end_time', time.time())

    # Calculate latency
    start_time = r.get('test_id:' + str(test_id) + ':start_time').decode()
    first_transcript_time = r.get('test_id:' + str(test_id) + ':first_transcript_time').decode()
    end_time = r.get('test_id:' + str(test_id) + ':end_time').decode()
    transcript_latency = first_transcript_time - start_time
    complete_and_ordered_latency = end_time - start_time
    # Save to redis
    r.set('test_id:' + str(test_id) + ':transcript_latency', transcript_latency)
    r.set('test_id:' + str(test_id) + ':complete_and_ordered_latency', complete_and_ordered_latency)

    
