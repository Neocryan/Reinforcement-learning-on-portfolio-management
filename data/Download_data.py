from __future__ import print_function
import numpy as np
import json, time, sys, csv
import pandas as pd
from datetime import datetime

if sys.version_info[0] == 3:
    from urllib.request import Request, urlopen
    from urllib.parse import urlencode
else:
    from urllib2 import Request, urlopen
    from urllib import urlencode

PUBLIC_COMMANDS = ['returnTicker', 'return24hVolume',
                   'returnOrderBook', 'returnTradeHistory',
                   'returnChartData', 'returnCurrencies',
                   'returnLoanOrders']

path = 'history/'

class Poloniex:
    def __init__(self):
        # Conversions
        self.timestamp_str = lambda timestamp=time.time(), \
        format="%Y-%m-%d %H:%M:%S": datetime.\
            fromtimestamp(timestamp).strftime(format)
        self.str_timestamp = lambda datestr=self.timestamp_str(), \
        format="%Y-%m-%d %H:%M:%S": int(time.mktime(time.strptime(datestr, \
                                                                  format)))
    #####################
    # Main Api Function #
    #####################
    def api(self, command = 'returnChartData', args={}):
        """
        returns 'False' if invalid command or if no APIKey or Secret is
        specified (if command is "private")
        returns {"error":"<error message>"} if API error
        """
        if command in PUBLIC_COMMANDS:
            url = 'https://poloniex.com/public?'
            args['command'] = command
            # valid: 5mins, 15mins, 30mins, 2h, 4h, 24h
            args['period'] = int(args['period']) * 60
            try:
                args['start'] = self.str_timestamp(args['start'])
                args['end'] = self.str_timestamp(args['end'])

            except:
                print('Error')
                return "Enter correc date format. E.g. 2015-01-01 00:00:00"
            result = {}
            for coin in ['BTC', 'ETH', 'XRP', 'LTC']:
                args['currencyPair'] = 'USDT_' + coin
                ret = urlopen(Request(url + urlencode(args)))
                print(url + urlencode(args))
                result[coin] = json.loads(ret.read().decode(encoding='UTF-8'))
                f = csv.writer(open(path + coin + '.csv', 'w', newline=''))
                f.writerow(['close', 'date', 'high','low', 'open',
                            'quoteVolume', 'volume', 'weightedAverage'])
                #print(path + coin + '.csv')
                for x in result[coin]:
                    f.writerow([x['close'], x['date'], x['high'], x['low'],
                                x['open'], x['quoteVolume'], x['volume'],
                                x['weightedAverage']])

            print("Data loaded.")
            return result

        else:
            return False

# BTC, ETH, XRP (Ripple), LTC (Litecoin)
