import time
from threading import Thread
from Models import *

TIME_INTERVAL = .0#86400# number of secs in 24 hours
TICKER = "AAPL"

model = DayTradeModel()
model.load()
def run_loop():
    while True:
        model.update_cached_offline()
        model.predict()
        time.sleep(TIME_INTERVAL)


def test_accuracy():
    with open(f'{TICKER}/info.json') as file:
        data = json.load(file)
    i = 200
    together_list = []
    last = 0
    last_predict = 0
    while i < 1000:
        prediction = model.predict() - last_predict
        last = data['Close'][i+1]-last
        last_predict = prediction

        down_together = last_predict[0][0]<0 and last<0
        up_together = last_predict[0][0] >0 and last>0

        together_list.append(down_together or up_together)
        i += 1
        time.sleep(TIME_INTERVAL)
    together_list = [int(a) for a in together_list]

    print(sum(together_list), len(together_list))
    print("percennt correct(in terms of going up or down)", sum(together_list)/len(together_list))

if __name__ == "__main__":
    # Create a new thread
    thread = Thread(target=run_loop)

    # Start the thread
    thread.start()
