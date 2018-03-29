import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)


def ani(i):
    try:
        graph_data = open('log.txt', 'r').read().split('//')
        weights = eval(graph_data[0])
        labels = 'Cash', 'BTC', 'ETH', 'LTC', 'XRP'
        explode = (0, 0, 0, 0, 0)
        rw = eval(graph_data[1])
        p1 = eval(graph_data[2])
        p2 = eval(graph_data[3])
        p3 = eval(graph_data[4])
        p4 = eval(graph_data[5])
        his = eval(graph_data[6])

        try:
            rw = rw[-100:]
            p1 = p1[-100:]
            p2 = p2[-100:]
            p3 = p3[-100:]
            p4 = p4[-100:]

        except:
            pass
        try:
            his = his[-1000:]
        except:
            pass
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax1.set_title('retrun in each term')
        ax2.set_title('ratio of each asset')
        ax3.set_title('assets price')
        ax4.set_title('total reward')
        ax2.pie(weights, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax2.axis('equal')
        ax1.plot(rw)

        ax3.plot(p1, 'r', alpha=0.3, label='BTC')
        ax3.plot(p2, 'g', alpha=0.3, label='ETH')
        ax3.plot(p3, 'navy', alpha=0.3, label='LTC')
        ax3.plot(p4, 'yellow', alpha=0.3, label='XRP')
        ax3.legend(loc=1)
        ax4.plot(his)
    except:
        print('json load fail')


anii = animation.FuncAnimation(fig, ani, interval=50)
plt.show()
