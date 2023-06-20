import time
from threading import Thread
from Models import *

TIME_INTERVAL = .0#86400# number of secs in 24 hours
TICKER = "AAPL"

model = DayTradeModel()
model.train()
def run_loop():
    while True:
        model.predict()
        time.sleep(TIME_INTERVAL)


def test_accuracy():
    with open(f'{TICKER}/info.json') as file:
        data = json.load(file)
    i = 200
    alll = []
    last = 0
    last_predict = 0
    while i < 1000:
        prediction = model.predict() - last_predict
        last = data['Close'][i+1]-last
        last_predict = prediction

        down_together = last_predict[0][0]<0 and last<0
        up_together = last_predict[0][0] >0 and last>0
        alll.append(down_together or up_together)
        i += 1
        time.sleep(TIME_INTERVAL)
    alll = [int(a) for a in alll]
    print(sum(alll), len(alll))
    print("percennt correct(in terms of going up or down)", sum(alll)/len(alll))

if __name__ == "__main__":
    # Create a new thread
    thread = Thread(target=run_loop)

    # Start the thread
    thread.start()
