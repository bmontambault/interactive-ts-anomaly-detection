import pandas as pd
import numpy as np
import fix_yahoo_finance as yf


def get_sp_tickers():
    return pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', header=0)[0]['Symbol'].values.tolist()


def get_data(ticker, start, end):
    
    daily = yf.download(ticker, start, end)['Close']
    daily = daily.rename(ticker)
    return daily


def get_all_data(tickers, start, end, first_ticker=None):
    
    all_data = []
    if first_ticker:
        tickers = tickers[tickers.index(first_ticker):]
    for ticker in tickers:
        try:
            data = get_data(ticker, start, end)
            all_data.append(data)
        except:
            print ('failed on ' + ticker)
    return pd.concat(all_data, axis=1)


def get_tech(start='2018-01-01', points=100):

    end = str(pd.to_datetime(start) + pd.Timedelta(days=points*2)).split()[0]
    tickers = ['CRM',
               'FB', 'AMZN', 'NFLX', 'GOOGL', 'MSFT',
               'AAPL', 'NVDA', 'INTC', 'DELL', 'IBM']
    data = get_all_data(tickers, start, end)
    data = data[:points]
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return data


def get_health(start='2018-01-01', points=100):

    end = str(pd.to_datetime(start) + pd.Timedelta(days=points*2)).split()[0]
    tickers = ['THC',
               'HCA', 'CNC', 'MOH', 'ANTM', 'PFE',
               'CYH', 'MDT', 'DGX', 'EHC', 'WCG']
    data = get_all_data(tickers, start, end)
    data = data[:points]
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return data

"""
def add_anomalies(data, n):
    
    rows, cols = data.shape
    anomaly_idx = np.random.choice(np.arange(rows), size=(cols,n))
    anomalies = np.random.choice([1,-1], size=(cols,10))*np.random.uniform(1,2,size=(cols,n))
    
    data_values = data.values
    for i in range(len(anomaly_idx)):
        data_values[anomaly_idx[i],i] += anomalies[i]
    new_data = pd.DataFrame(data_values)
    new_data.columns = data.columns
    return new_data
"""

"""
if __name__ == "__main__":
    
    tech_data = get_tech()
    health_data = get_health()
    
    tech_data1 = tech_data[:100]
    tech_data2 = tech_data[100:]
    
    health_data1 = health_data[:100]
    health_data2 = health_data[100:]
    
    np.random.seed(34432)
    tech_data1_anomalies = add_anomalies(tech_data1, 10)
    tech_data2_anomalies = add_anomalies(tech_data2, 10)
    health_data1_anomalies = add_anomalies(health_data1, 10)
    health_data2_anomalies = add_anomalies(health_data2, 10)
    
    tech_data1.to_csv('tech_data1.csv')
    tech_data1_anomalies.to_csv('tech_data1_anomalies.csv')
    tech_data2_anomalies.to_csv('tech_data2_anomalies.csv')
    health_data1.to_csv('health_data1.csv')
    health_data1_anomalies.to_csv('health_data1_anomalies.csv')
    health_data2_anomalies.to_csv('health_data2_anomalies.csv')
"""