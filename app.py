
from flask import Flask, request, jsonify, send_from_directory, render_template
import json
import pickle
import pandas as pd 
import random
from scipy.stats import gaussian_kde
from scipy.signal import decimate
from dtaidistance import dtw

from multiprocessing import Pool, cpu_count

from datetime import datetime

#from settings import exe_list, input_type, decimated_signals
#from settings import  model_folder, model_out, ver
#from model_functions import apply_decimator

ver = '1.4'

decimated_signals = True

app = Flask(__name__, static_folder='static')

with open('MODEL/' + 'feature_vector_result_1.4.pkl', 'rb') as f:
    feature_vector_result = pd.read_pickle(f)

with open('MODEL/' + f"reference_ts_bvh_{ver}.pkl", 'rb') as f:
    reference_ts_bvh = pd.read_pickle(f)

with open('MODEL/' + f"reference_ts_bvh_decim_{ver}.pkl", 'rb') as f:
    reference_ts_bvh_decim = pd.read_pickle(f)

with open('MODEL/' + f'decim_bvh_{ver}.json') as json_data:
    decim_coeff = json.load(json_data)
    json_data.close()

decim_coeff = decim_coeff['value']

reference_ts_mtx_decim_df = pd.read_pickle('MODEL/'  + f"reference_ts_bvh_decim_{ver}.pkl")

reference_ts_mtx_df = pd.read_pickle('MODEL/'  + f"reference_ts_bvh_{ver}.pkl")

with open('MODEL/' + f'feature_vector_bvh_{ver}.json') as json_data:
    features = json.load(json_data)
    json_data.close()

features = features['features']

with open('MODEL/' + 'C249e_D251_DD_1_bvh_exe_file_dict.json', 'rb') as json_data:
    json_exe_dict = json.load(json_data)
    json_data.close()

with open('MODEL/' +  f'th_dict_{ver}.json', 'rb') as json_data:
    th_dict = json.load(json_data)
    json_data.close()

with open('MODEL/' +  f'k_best_components_bvh_{ver}.json') as json_data:
    k_best_components = json.load(json_data)
    json_data.close()

####################################################################################

def process_data_kde(data, test_value):
    # Calculate the likelihood of the test value
    kde = gaussian_kde(data)
    likelihood = kde.evaluate(test_value)

    return {"likelihood": likelihood[0]*100.0}

def calc_dictance(data1, data2):
    data_df = pd.concat([data1, data2], axis=1, ignore_index=True)
    data_df = data_df.ffill()

    return  dtw.distance(list(data_df[data_df.columns[0]]), list(data_df[data_df.columns[1]]))

def compute_distance(args):
    key, tested_df = args
    temp_list = []
    for ang in features:
        if decimated_signals:
            decim_signal = apply_decimator(list(tested_df[ang]), decim_coeff )
            temp = dtw.distance(list(decim_signal), list(reference_ts_mtx_decim_df.loc[key][ang]))
        else:
            temp = dtw.distance(list(tested_df[ang]), list(reference_ts_mtx_df.loc[key][ang]))
        temp_list.append(temp)
    return temp_list

def apply_decimator(signal, decimation_factor):
    if decimation_factor > 0.0:
        decimated_signal = decimate(signal, int(decimation_factor))
    else:
        decimated_signal = signal
    return decimated_signal

####################################################################################

@app.route('/')
def index():
    print('index')
    return render_template('endpoints.html')

@app.route('/process_mle', methods=['GET', 'POST'])
def process_json():
    # Get JSON data from the request
    data = request.get_json()
    #print(data)

    test_file_name = data.get('test_file_name')
    print(test_file_name)

    exe_label = data.get('label','1')
    print(exe_label)

    with open(test_file_name, 'rb') as json_data:
        test_record = json.load(json_data)
        json_data.close()

    start_time = datetime.now()

    syn_dist = 0    
    for ch in k_best_components[exe_label]:
        dist = calc_dictance(reference_ts_bvh[ch][exe_label], pd.Series(test_record[ch]))
        #print(dist)
        syn_dist = syn_dist + dist
        print(f"component distance: {round(dist,1)}")
    print(f"total distance: {round(syn_dist,1)}")

    MLE = process_data_kde(feature_vector_result[exe_label], syn_dist)
    th_level = data.get('TH')
    if th_level == 'auto':
        th_value = th_dict[exe_label]
    else:
        th_value = feature_vector_result[exe_label].quantile(float(th_level))

    MLE_TH = process_data_kde(feature_vector_result[exe_label], th_value)

    MLE_value = round(MLE['likelihood'],4)

    MLE_TH_value = round(MLE_TH['likelihood'],4)

    print(f'MLE: {MLE_value}')

    print(k_best_components[exe_label])

    end_time = datetime.now()

    delta = end_time - start_time

    processed_data = {'MLE': MLE_value, 
                      'MLE_TH': MLE_TH_value,
                      'Processing time': round(delta.total_seconds(), 1), 
                      'label': exe_label,
                      'test_file_name': test_file_name,
                      'label_right': 1 if MLE_value > MLE_TH_value else 0}

    return jsonify(processed_data)

####################################################################################
@app.route('/process_random_record', methods=['POST', 'GET'])
def process_data():
    test_file_name = request.args.get('test_file_name')
    print(test_file_name)

    exe_label = request.args.get('label','1')
    print(exe_label)

    with open(test_file_name, 'rb') as json_data:
        test_record = json.load(json_data)
        json_data.close()

    start_time = datetime.now()

    syn_dist = 0    
    for ch in k_best_components[exe_label]:
        dist = calc_dictance(reference_ts_bvh[ch][exe_label], pd.Series(test_record[ch]))
        #print(dist)
        syn_dist = syn_dist + dist
        print(f"component distance: {round(dist,1)}")
    print(f"total distance: {round(syn_dist,1)}")

    MLE = process_data_kde(feature_vector_result[exe_label], syn_dist)

    MLE_value = round(MLE['likelihood'],3)

    print(f'MLE: {MLE_value}')

    end_time = datetime.now()

    delta = end_time - start_time

    processed_data = {'MLE': MLE_value, 
                      'Processing time': round(delta.total_seconds(), 1), 
                      'label': exe_label,
                      'test_file_name': test_file_name}

    return jsonify(processed_data)

####################################################################################
@app.route('/process_json_decim', methods=['POST', 'GET'])
def process_decim():

    data = request.get_json()

    mode = data.get('mode', '1')
    multi_cpu = data.get('multi_cpu', '0')
    test_file_name = 'data_mode'
    if mode == 'link':
        test_file_name = data.get('test_file_name')
        with open(test_file_name, 'rb') as json_data:
            test_record = json.load(json_data)
            json_data.close()
    elif mode == 'data':
        test_record = data.get('data')
    elif mode == '1':
        print('bad request')

    start_time = datetime.now()
    
    tested_df = pd.DataFrame(test_record)

    if multi_cpu == '0':
        exe_test_result = []
        for key in json_exe_dict.keys():
            temp_list = []
            for ang in features:
                if decimated_signals:
                    decim_signal = apply_decimator(list(tested_df[ang]), decim_coeff )
                    temp = dtw.distance(list(decim_signal), list(reference_ts_mtx_decim_df.loc[key][ang]))
                else:
                    temp = dtw.distance(list(tested_df[ang]), list(reference_ts_mtx_df.loc[key][ang]))
                temp_list.append(temp)
            exe_test_result.append(temp_list)
    
    elif multi_cpu == '1':
        args_list = [(key, tested_df) for key in json_exe_dict.keys()]
                
        with Pool(cpu_count()-1) as p:
            exe_test_result = p.map(compute_distance, args_list)
   
    one_exe_tested_df = pd.DataFrame(exe_test_result, index=json_exe_dict.keys(), 
                                     columns=features)
     
    exe_label = one_exe_tested_df.sum(axis=1).idxmin()

    end_time = datetime.now()

    delta = end_time - start_time

    n_cpu = cpu_count() - 1
    if multi_cpu == '0':
        n_cpu = 1
        
    processed_data = {'Processing time': round(delta.total_seconds(), 1), 
                      'label': exe_label,
                      'test_file_name': test_file_name,
                      'n_cpu': n_cpu}
    return jsonify(processed_data)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
