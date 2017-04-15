from twython import TwythonStreamer
from queue import Queue
import threading
import csv
import time

class Listener(TwythonStreamer):
    q = Queue()

    def on_success(self, data):
        if 'text' in data:
            self.q.put(data['text'])

    def on_error(self, status_code, data):
        print(status_code)


def credentials():
    return ("******")


def main(track):
    APP_KEY, APP_SECRET, OAUTH_TOKEN, OAUTH_SECRET = credentials()

    l = Listener(APP_KEY, APP_SECRET, OAUTH_TOKEN, OAUTH_SECRET)
    l_thread = threading.Thread(target=l.statuses.filter, kwargs={'track': track})
    l_thread.start()

    while True:
        try:
            with open(''.join([track, "tweets", ".csv"]), 'a') as f:
                writer = csv.writer(f, delimiter=' ')
                writer.writerow([l.q.get()])
        except Exception:
            pass
        time.sleep(0.03)