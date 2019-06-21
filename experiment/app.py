from flask import Flask,render_template,redirect,url_for,request
import os
import uuid
import pandas as pd
import numpy as np
import json
from datetime import datetime

import utils

path=os.path.dirname(os.path.realpath(__file__))

app=Flask(__name__)
with open(path + '/secret_key.txt') as f:
    app.secret_key= f.read()
    


@app.route('/consent/<experiment>/<participant_id>/<task>/<function_set>/<trial>')
def consent(experiment, participant_id, task, function_set, trial):
    
    css_path = '../../../../../..'
    return render_template('consent.html',
                           experiment=experiment,
                           participant_id=participant_id,
                           task=task,
                           function_set=function_set,
                           trial=trial,
                           css_path=css_path)


@app.route('/experiment_1_instructions/<experiment>/<participant_id>/<task>/<function_set>/<trial>')
def experiment_1_instructions(experiment, participant_id, task, function_set, trial):
    
    css_path = '../../../../../..'
    return render_template('experiment_1_instructions.html',
                           experiment=experiment,
                           participant_id=participant_id,
                           task=task,
                           function_set=function_set,
                           trial=trial,
                           css_path=css_path)
    
@app.route('/experiment_2_instructions/<experiment>/<participant_id>/<task>/<function_set>/<trial>')
def experiment_2_instructions(experiment, participant_id, task, function_set, trial):
    
    css_path = '../../../../../..'
    return render_template('experiment_2_instructions.html',
                           experiment=experiment,
                           participant_id=participant_id,
                           task=task,
                           function_set=function_set,
                           trial=trial,
                           css_path=css_path)
    
    
@app.route('/experiment_3_instructions/<experiment>/<participant_id>/<task>/<function_set>/<trial>')
def experiment_3_instructions(experiment, participant_id, task, function_set, trial):
    
    css_path = '../../../../../..'
    return render_template('experiment_3_instructions.html',
                           experiment=experiment,
                           participant_id=participant_id,
                           task=task,
                           function_set=function_set,
                           trial=trial,
                           css_path=css_path)
    


@app.route('/predict_instructions/<experiment>/<participant_id>/<task>/<function_set>/<trial>')
def predict_instructions(experiment, participant_id, task, function_set, trial):
    
    css_path = '../../../../../..'
    with open(path + '/config.json', 'r') as f:
        config = json.load(f)
    point_size = config["point_size"]
    plot_height = config["plot_height"]
    plot_width = config["plot_width"]
    padding = config["padding"]
    train_size = config["train_size"]
    test_size = config["test_size"]
    
    if int(experiment) < 3:
        data_path = path + '/../data/{}_sample.csv'.format(['rq','sm_lin'][int(function_set)])
    else:
        data_path = path + '/../data/{}_sample.csv'.format(['tech','health'][int(function_set)])
    
    data = pd.read_csv(data_path)
    y = data.values[:,int(trial)]
    scaled_y = utils.scale_function(np.array(y), padding, plot_height-padding).tolist()
    X = np.arange(plot_width, step=plot_width/len(y)).tolist()
    
    trainX, trainy, testX, testy = utils.get_series_extrap(X, scaled_y, train_size, test_size)
    
    return render_template('predict.html',
                           experiment=experiment,
                           participant_id=participant_id,
                           task=task,
                           function_set=function_set,
                           trial=trial,
                           point_size=point_size,
                           plot_height=plot_height,
                           plot_width=plot_width,
                           padding=padding,
                           trainX=trainX,
                           trainy=trainy,
                           testX=testX,
                           testy=testy,
                           instructions=True,
                           css_path=css_path)
    

@app.route('/predict_instructions2/<experiment>/<participant_id>/<task>/<function_set>/<trial>')
def predict_instructions2(experiment, participant_id, task, function_set, trial):
    
    css_path = '../../../../../..'
    
    return render_template('predict_instructions2.html',
                           experiment=experiment,
                           participant_id=participant_id,
                           task=task,
                           function_set=function_set,
                           trial=trial,
                           css_path=css_path)
    

@app.route('/predict/<experiment>/<participant_id>/<task>/<function_set>/<trial>')
def predict(experiment, participant_id, task, function_set, trial):
    
    css_path = '../../../../../..'
    
    with open(path + '/config.json', 'r') as f:
        config = json.load(f)
    point_size = config["point_size"]
    train_size = config["train_size"]
    test_size = config["test_size"]
    plot_height = config["plot_height"]
    plot_width = config["plot_width"]
    padding = config["padding"]
    
    data_path = path + '/../data/{}{}.csv'.format(['rq_task','sm_lin_task'][int(function_set)],task)
        
    data = pd.read_csv(data_path)
    y = data.values[:,int(trial)]
    scaled_y = utils.scale_function(np.array(y), padding, plot_height-padding).tolist()
    X = np.arange(plot_width, step=plot_width/len(y)).tolist()
    
    trainX, trainy, testX, testy = utils.get_series_extrap(X, scaled_y, train_size, test_size)
    
    return render_template('predict.html', trial=trial, trainX=trainX,
                           trainy=trainy, testX=testX, testy=testy,
                           plot_height=plot_height,
                           point_size=point_size,
                           plot_width=plot_width,
                           task=task,
                           experiment=experiment,
                           participant_id=participant_id,
                           function_set=function_set,
                           instructions=False,
                           css_path=css_path)


@app.route('/detect_instructions/<experiment>/<participant_id>/<task>/<function_set>/<trial>')
def detect_instructions(experiment, participant_id, task, function_set, trial):
    
    css_path = '../../../../../..'
    with open(path + '/config.json', 'r') as f:
        config = json.load(f)
    point_size = config["point_size"]
    plot_height = config["plot_height"]
    plot_width = config["plot_width"]
    padding = config["padding"]
    train_size = config["train_size"]
    test_size = config["test_size"]
    
    if int(experiment) < 3:
        data_path = path + '/../data/{}_sample_anomalies.csv'.format(['rq','sm_lin'][int(function_set)])
    else:
        data_path = path + '/../data/{}_sample_anomalies.csv'.format(['tech','health'][int(function_set)])
    
    data = pd.read_csv(data_path)
    y = data.values[:,int(trial)]
    scaled_y = utils.scale_function(np.array(y), padding, plot_height-padding).tolist()
    X = np.arange(plot_width, step=plot_width/len(y)).tolist()
    
    trainX, trainy, testX, testy = utils.get_series_extrap(X, scaled_y, train_size, test_size)
    
    return render_template('detect.html',
                           experiment=experiment,
                           participant_id=participant_id,
                           task=task,
                           function_set=function_set,
                           trial=trial,
                           point_size=point_size,
                           plot_height=plot_height,
                           plot_width=plot_width,
                           padding=padding,
                           trainX=trainX,
                           trainy=trainy,
                           testX=testX,
                           testy=testy,
                           instructions=True,
                           css_path=css_path)


@app.route('/detect_instructions2/<experiment>/<participant_id>/<task>/<function_set>/<trial>')
def detect_instructions2(experiment, participant_id, task, function_set, trial):
    
    return render_template('detect_instructions2.html', experiment=experiment,
                           participant_id=participant_id, task=task,
                           function_set=function_set, trial=trial)


@app.route('/detect/<experiment>/<participant_id>/<task>/<function_set>/<trial>')
def detect(experiment, participant_id, task, function_set, trial):
    
    css_path = '../../../../../..'
    
    with open(path + '/config.json', 'r') as f:
        config = json.load(f)
    point_size = config["point_size"]
    train_size = config["train_size"]
    test_size = config["test_size"]
    plot_height = config["plot_height"]
    plot_width = config["plot_width"]
    padding = config["padding"]
    
    if int(experiment) < 3:
        data_path = path + '/../data/{}{}_anomalies.csv'.format(['rq_task','sm_lin_task'][int(function_set)],task)
    else:
        data_path = path + '/../data/{}{}_anomalies.csv'.format(['health_task', 'tech_task'][int(function_set)],task)
    
    print (data_path)
    data = pd.read_csv(data_path)
    y = data.values[:,int(trial)]
    scaled_y = utils.scale_function(np.array(y), padding, plot_height-padding).tolist()
    X = np.arange(plot_width, step=plot_width/len(y)).tolist()
    
    trainX, trainy, testX, testy = utils.get_series_extrap(X, scaled_y, train_size, test_size)
    
    return render_template('detect.html', trial=trial, trainX=trainX,
                           trainy=trainy, testX=testX, testy=testy,
                           plot_height=plot_height,
                           point_size=point_size,
                           plot_width=plot_width,
                           task=task,
                           experiment=experiment,
                           participant_id=participant_id,
                           function_set=function_set,
                           instructions=False,
                           css_path=css_path)


@app.route('/demographics/<experiment>/<participant_id>')
def demographics(experiment, participant_id):
    
    return render_template('demographics.html', experiment=experiment, participant_id=participant_id)


@app.route('/debrief/<experiment>/<participant_id>')
def debrief(experiment, participant_id):
    
    return render_template('debrief.html', experiment=experiment, 
                           participant_id=participant_id)


@app.route('/experiment_1')
def experiment_1():
    
    participant_id = str(uuid.uuid4())
    return redirect(url_for('consent', experiment=1,
                            participant_id=participant_id,
                            task=1, function_set=0, trial=0,
                            ))
    
    
@app.route('/experiment_2')
def experiment_2():
    
    participant_id = str(uuid.uuid4())
    return redirect(url_for('consent', experiment=2,
                            participant_id=participant_id,
                            task=1, function_set=0, trial=0,
                            ))
    

@app.route('/experiment_3')
def experiment_3():
    
    participant_id = str(uuid.uuid4())
    return redirect(url_for('consent', experiment=3,
                            participant_id=participant_id,
                            task=1, function_set=0, trial=0,
                            ))
    

@app.route('/data', methods=['GET','POST'])
def data():
    
    d = request.json
    with open(f"results/{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.json", 'w') as f:
        json.dump(d, f)
    return ''
    

if __name__=="__main__":
    app.run()    
    