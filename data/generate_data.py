import numpy as np
import pandas as pd

from artificial import rq_samples, sm_lin_samples
from stocks import get_health, get_tech


def add_anomalies(data, n):
    
    if len(data.shape) == 1:
        rows = data.shape[0]
        anomaly_idx = np.random.choice(np.arange(rows), size=n)
        anomalies = (np.random.choice([1,-1], size=n)*np.random.uniform(1,2,size=n))
        data_values = data.values
        
        data_values[anomaly_idx] += anomalies
        new_data = pd.DataFrame(data_values)
    
    else:
        
        rows, cols = data.shape
        anomaly_idx = np.random.choice(np.arange(rows), size=(cols,n))
        anomalies = (np.random.choice([1,-1], size=(cols,n))*np.random.uniform(1,2,size=(cols,n)))
        
        data_values = data.values
        for i in range(len(anomaly_idx)):
            data_values[anomaly_idx[i],i] += anomalies[i]
        new_data = pd.DataFrame(data_values)
        new_data.columns = data.columns
    return new_data


if __name__ == "__main__":
    
    nanomalies=10
    seed=34324
    
    rq_data = pd.DataFrame(rq_samples()[1].T)
    sm_lin_data = pd.DataFrame(sm_lin_samples()[1].T)
    tech_data = get_tech()
    health_data = get_health()
    
    rq_sample_col = rq_data.columns[0]
    rq_task1_cols = rq_data.columns[1:6]
    rq_task2_cols = rq_data.columns[6:]
    
    sm_lin_sample_col = sm_lin_data.columns[0]
    sm_lin_task1_cols = sm_lin_data.columns[1:6]
    sm_lin_task2_cols = sm_lin_data.columns[6:]
    
    tech_sample_col = tech_data.columns[0]
    tech_task1_cols = tech_data.columns[1:6]
    tech_task2_cols = tech_data.columns[6:]
    
    health_sample_col = health_data.columns[0]
    health_task1_cols = health_data.columns[1:6]
    health_task2_cols = health_data.columns[6:]
    
    rq_sample = rq_data[rq_sample_col]
    rq_task1 = rq_data[rq_task1_cols]
    rq_task2 = rq_data[rq_task2_cols]
    
    sm_lin_sample = sm_lin_data[sm_lin_sample_col]
    sm_lin_task1 = sm_lin_data[sm_lin_task1_cols]
    sm_lin_task2 = sm_lin_data[sm_lin_task2_cols]
    
    tech_sample = tech_data[tech_sample_col]
    tech_task1 = tech_data[tech_task1_cols]
    tech_task2 = tech_data[tech_task2_cols]
    
    health_sample = health_data[health_sample_col]
    health_task1 = health_data[health_task1_cols]
    health_task2 = health_data[health_task2_cols]
    
    np.random.seed(seed)
    rq_sample_anomalies = add_anomalies(rq_sample.copy(), nanomalies)
    rq_task1_anomalies = add_anomalies(rq_task1.copy(), nanomalies)
    rq_task2_anomalies = add_anomalies(rq_task2.copy(), nanomalies)
    
    sm_lin_sample_anomalies = add_anomalies(sm_lin_sample.copy(), nanomalies)
    sm_lin_task1_anomalies = add_anomalies(sm_lin_task1.copy(), nanomalies)
    sm_lin_task2_anomalies = add_anomalies(sm_lin_task2.copy(), nanomalies)
    
    tech_sample_anomalies = add_anomalies(tech_sample.copy(), nanomalies)
    tech_task1_anomalies = add_anomalies(tech_task1.copy(), nanomalies)
    tech_task2_anomalies = add_anomalies(tech_task2.copy(), nanomalies)
    
    health_sample_anomalies = add_anomalies(health_sample.copy(), nanomalies)
    health_task1_anomalies = add_anomalies(health_task1.copy(), nanomalies)
    health_task2_anomalies = add_anomalies(health_task2.copy(), nanomalies)
    
    rq_sample.to_csv('../experiment/series/rq_sample.csv',index=False)
    rq_task1.to_csv('../experiment/series/rq_task1.csv',index=False)
    rq_task2.to_csv('../experiment/series/rq_task2.csv',index=False)
    
    sm_lin_sample.to_csv('../experiment/series/sm_lin_sample.csv',index=False)
    sm_lin_task1.to_csv('../experiment/series/sm_lin_task1.csv',index=False)
    sm_lin_task2.to_csv('../experiment/series/sm_lin_task2.csv',index=False)
    
    tech_sample.to_csv('../experiment/series/tech_sample.csv',index=False)
    tech_task1.to_csv('../experiment/series/tech_task1.csv',index=False)
    tech_task2.to_csv('../experiment/series/tech_task2.csv',index=False)
    
    health_sample.to_csv('../experiment/series/health_sample.csv',index=False)
    health_task1.to_csv('../experiment/series/health_task1.csv',index=False)
    health_task2.to_csv('../experiment/series/health_task2.csv',index=False)
    
    rq_sample_anomalies.to_csv('../experiment/series/rq_sample_anomalies.csv',index=False)
    rq_task1_anomalies.to_csv('../experiment/series/rq_task1_anomalies.csv',index=False)
    rq_task2_anomalies.to_csv('../experiment/series/rq_task2_anomalies.csv',index=False)
    
    sm_lin_sample_anomalies.to_csv('../experiment/series/sm_lin_sample_anomalies.csv',index=False)
    sm_lin_task1_anomalies.to_csv('../experiment/series/sm_lin_task1_anomalies.csv',index=False)
    sm_lin_task2_anomalies.to_csv('../experiment/series/sm_lin_task2_anomalies.csv',index=False)
    
    tech_sample_anomalies.to_csv('../experiment/series/tech_sample_anomalies.csv',index=False)
    tech_task1_anomalies.to_csv('../experiment/series/tech_task1_anomalies.csv',index=False)
    tech_task2_anomalies.to_csv('../experiment/series/tech_task2_anomalies.csv',index=False)
    
    health_sample_anomalies.to_csv('../experiment/series/health_sample_anomalies.csv',index=False)
    health_task1_anomalies.to_csv('../experiment/series/health_task1_anomalies.csv',index=False)
    health_task2_anomalies.to_csv('../experiment/series/health_task2_anomalies.csv',index=False)
