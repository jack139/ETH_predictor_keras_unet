# coding:utf-8
import sys
import urllib3, json, base64, time, hashlib
from urllib3.contrib.socks import SOCKSProxyManager
from datetime import datetime

urllib3.disable_warnings()

# okex 提供的接入参数
Passphrase = ""
apikey = ""
secretkey = ""

method = 'GET'
host = 'https://www.okex.com'


# 
def get_data(uri, instrument_id, num): 

    unixtime = int(time.time())

    sign_str = '%d%s%s%s' % (unixtime, method, uri, secretkey)

    # 签名
    sha256 = hashlib.sha256(sign_str.encode('utf-8')).hexdigest().encode('utf-8')
    signature_str =  base64.b64encode(sha256).decode('utf-8')

    #print(sign_str.encode('utf-8'))
    #print(sha256)
    #print(signature_str)

    headers ={
        'OK-ACCESS-KEY' : apikey,
        'OK-ACCESS-SIGN' : signature_str,
        'OK-ACCESS-TIMESTAMP' : '%d'%unixtime,
        'OK-ACCESS-PASSPHRASE' : Passphrase,
        'Accept' : 'application/json',
        'Content-Type' : 'application/json; charset=UTF-8',
    }

    # 发起调用
    url = host+uri

    print(url)

    start_time = datetime.now()
    # 不使用代理
    #pool = urllib3.PoolManager(num_pools=2, timeout=180, retries=False) 
    #r = pool.urlopen(method, url, headers=headers)
    # 使用sock5代理
    proxy = SOCKSProxyManager('socks5h://localhost:1080/') 
    r = proxy.request(method, url, headers = headers)
    print('[Time taken: {!s}]'.format(datetime.now() - start_time))

    #print(r.status)
    #print(r.data)
    if r.status!=200:
        print("fail: ", r.status, r.data)
        return []

    # 获取数据，截断为num， 注意顺序：最新数据在前
    data = json.loads(r.data.decode('utf-8'))
    data = data[:num]

    return data

# instrument_id = 'BTC-USDT' 或 'ETH-USDT'
def get_recent(csvpath, instrument_id='ETH-USDT', gap=3600, num=128): 
    # 获取当前数据 200条
    uri = '/api/spot/v3/instruments/%s/candles'%instrument_id
    param = 'granularity=%d'%gap

    url = '%s?%s'%(uri, param)

    X = get_data(url, instrument_id, num)

    with open(csvpath, 'w') as f:
        f.write("time,open,high,low,close,volume\n")
        for x in X[::-1]: # 逆序
            s = ','.join(x)
            f.write('%s\n'%s)

    return X

# instrument_id = 'BTC-USDT' 或 'ETH-USDT'
def get_history(csvpath, instrument_id='ETH-USDT', gap=3600, num=300): 

    # 1年的数据
    span = [
        ('2020-01-10T00:00:00.000Z', '2019-12-30T01:00:00.000Z'),
        ('2020-01-20T00:00:00.000Z', '2020-01-10T01:00:00.000Z'),
        ('2020-01-30T00:00:00.000Z', '2020-01-20T01:00:00.000Z'),

        ('2020-02-10T00:00:00.000Z', '2020-01-30T01:00:00.000Z'),
        ('2020-02-20T00:00:00.000Z', '2020-02-10T01:00:00.000Z'),
        ('2020-02-28T00:00:00.000Z', '2020-02-20T01:00:00.000Z'), # 注意 2月28日

        ('2020-03-10T00:00:00.000Z', '2020-02-28T01:00:00.000Z'), # 注意 2月28日
        ('2020-03-20T00:00:00.000Z', '2020-03-10T01:00:00.000Z'),
        ('2020-03-30T00:00:00.000Z', '2020-03-20T01:00:00.000Z'),

        ('2020-04-10T00:00:00.000Z', '2020-03-30T01:00:00.000Z'),
        ('2020-04-20T00:00:00.000Z', '2020-04-10T01:00:00.000Z'),
        ('2020-04-30T00:00:00.000Z', '2020-04-20T01:00:00.000Z'),

        ('2020-05-10T00:00:00.000Z', '2020-04-30T01:00:00.000Z'),
        ('2020-05-20T00:00:00.000Z', '2020-05-10T01:00:00.000Z'),
        ('2020-05-30T00:00:00.000Z', '2020-05-20T01:00:00.000Z'),

        ('2020-06-10T00:00:00.000Z', '2020-05-30T01:00:00.000Z'),
        ('2020-06-20T00:00:00.000Z', '2020-06-10T01:00:00.000Z'),
        ('2020-06-30T00:00:00.000Z', '2020-06-20T01:00:00.000Z'),

        ('2020-07-10T00:00:00.000Z', '2020-06-30T01:00:00.000Z'),
        ('2020-07-20T00:00:00.000Z', '2020-07-10T01:00:00.000Z'),
        ('2020-07-30T00:00:00.000Z', '2020-07-20T01:00:00.000Z'),

        ('2020-08-10T00:00:00.000Z', '2020-07-30T01:00:00.000Z'),
        ('2020-08-20T00:00:00.000Z', '2020-08-10T01:00:00.000Z'),
        ('2020-08-30T00:00:00.000Z', '2020-08-20T01:00:00.000Z'),

        ('2020-09-10T00:00:00.000Z', '2020-08-30T01:00:00.000Z'),
        ('2020-09-20T00:00:00.000Z', '2020-09-10T01:00:00.000Z'),
        ('2020-09-30T00:00:00.000Z', '2020-09-20T01:00:00.000Z'),

        ('2020-10-10T00:00:00.000Z', '2020-09-30T01:00:00.000Z'),
        ('2020-10-20T00:00:00.000Z', '2020-10-10T01:00:00.000Z'),
        ('2020-10-30T00:00:00.000Z', '2020-10-20T01:00:00.000Z'),

        ('2020-11-10T00:00:00.000Z', '2020-10-30T01:00:00.000Z'),
        ('2020-11-20T00:00:00.000Z', '2020-11-10T01:00:00.000Z'),
        ('2020-11-30T00:00:00.000Z', '2020-11-20T01:00:00.000Z'),

        ('2020-12-10T00:00:00.000Z', '2020-11-30T01:00:00.000Z'),
        ('2020-12-20T00:00:00.000Z', '2020-12-10T01:00:00.000Z'),
        ('2020-12-30T00:00:00.000Z', '2020-12-20T01:00:00.000Z'),

        ('2021-01-10T00:00:00.000Z', '2020-12-30T01:00:00.000Z'),
        ('2021-01-20T00:00:00.000Z', '2021-01-10T01:00:00.000Z'),
    ]

    # 获取历史数据 每次返回 300条
    uri = '/api/spot/v3/instruments/%s/history/candles'%instrument_id

    f =  open(csvpath, 'w')
    f.write("time,open,high,low,close,volume\n")

    all = []

    for t in span:
        param = 'start=%s&end=%s&granularity=%d'%(t[0], t[1], gap)

        url = '%s?%s'%(uri, param)

        X = get_data(url, instrument_id, num)
    
        for x in X[::-1]: # 逆序
            s = ','.join(x)
            f.write('%s\n'%s)

        all.extend(X)

        print(len(X))

    f.close()

    return all


if __name__ == '__main__':
    # 取最近的
    X = get_recent('../dataset/eth_now.csv', 'ETH-USDT')
    print(len(X))
    # 取历史数据
    #X = get_history('../dataset/eth_history.csv', 'ETH-USDT')
    #print(len(X))
