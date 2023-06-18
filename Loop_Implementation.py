import time, datetime
from threading import Thread
from Models import *

time_interval = .0#86400# number of secs in 24 hours
ticker = "AAPL"

model = DayTradeModel()
#model.load()
model.train()
def run_loop():
    with open(f'{ticker}/info.json') as file:
        data = json.load(file)
    i = 200
    alll = []
    l = 0
    l_p = 0
    info_keys = model.information_keys
    while i < 1000:
        b = model.predict() - l_p
        l = data['Close'][i+1]-l
        l_p = b

        a = l_p[0][0]<0 and l<0
        b = l_p[0][0] >0 and l>0
        alll.append(a or b)
        i += 1
        time.sleep(time_interval)
    alll = [int(a) for a in alll]
    print(alll[1:10])
    print(sum(alll), len(alll))
    print(sum(alll)/len(alll))

if __name__ == "__main__":
    # Create a new thread
    thread = Thread(target=run_loop)

    # Start the thread
    thread.start()
