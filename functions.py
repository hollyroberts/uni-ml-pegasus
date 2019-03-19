# This file contains functions used in the coursework
# It's easier to separate them out for readability

import matplotlib.pyplot as plt
import matplotlib as mlp
from pylab import MaxNLocator

def format_acc(number, convert_to_percentage=False):
    if convert_to_percentage:
        return f"{number * 100:,.2f}%"
    else:
        return f"{number:,.2f}"

def format_time(duration):
    seconds = duration % 60
    duration //= 60
    minutes = int(duration % 60)
    hours = int(duration // 60)

    if hours > 0:
        time_string = f"{hours}:{minutes:02}:{seconds:02.0f}"
    elif minutes > 0:
        time_string = f"{minutes:02}:{seconds:02.0f}"
    else:
        time_string = f"{seconds:.1f}s"

    return time_string

def create_end_graphs(acc, val_acc, loss, val_loss):
    plt.figure(figsize=(10, 4))

    sp = plt.subplot(1, 2, 1)
    # noinspection PyUnresolvedReferences
    sp.yaxis.set_major_formatter(mlp.ticker.StrMethodFormatter('{x}%'))
    sp.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.plot(acc, 'b-', label='training')
    plt.plot(val_acc, 'g-', label='test')
    plt.legend(loc='lower right')

    sp = plt.subplot(1, 2, 2)
    plt.title("Loss")
    plt.xlabel("Epoch")
    sp.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(loss, 'b-', label='training')
    plt.plot(val_loss, 'g-', label='test')
    plt.legend(loc='upper right')

    plt.show()
