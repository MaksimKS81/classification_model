#import unittest
#from rest_ui import app
import json
import requests
import pandas as pd

from datetime import datetime

#from settings import ver

from sklearn.metrics import precision_recall_fscore_support

test_out = "REST_UI_TEST_OUT/"

th_level = 'auto' # or 0.95 

test_decim = True

server_mode = 'data' # 'link' or 'data'

test_mode = 'remote' # 'local' or 'remote'

if test_mode == 'local':
    server_url = "http://127.0.0.1:8080/process_json_decim"
elif test_mode == 'remote':
    server_url = 'https://classification-model-git-nh5zsg5ciq-uc.a.run.app/process_json_decim'
    

multi_cpu = '1'

with open('MODEL/' + 'C249e_D251_DD_1_bvh_exe_file_dict.json', 'rb') as json_data:
    json_exe_dict = json.load(json_data)
    json_data.close()
    
def counted_values(data):
    counted_value = {'0':0,'1':0}
    for i in list(data):
        if i == 0: counted_value['0'] +=1
        elif i == 1: counted_value['1'] += 1
    return counted_value

class TestAPI():
    def setUp(self):
        #self.app = app.test_client()
        with open('MODEL/' + 'C249e_D251_DD_1_bvh_exe_file_dict.json', 'rb') as json_data:
            json_exe_dict = json.load(json_data)
            json_data.close()
        self.json_exe_dict = json_exe_dict
        
    def test_process_json(self, url, test_data):       

        # Send a POST request to the /process_json endpoint with the test data
        response = requests.post(url, json=test_data, timeout=30)

        # Check if the status code of the response is 200
        if response.status_code == 200:
            print(response.json())
            return response.json()
        else:
        # If the request was not successful, raise an exception
            raise Exception(f"Request failed with status code {response.status_code}")
        # Check if the response data is correct
        #self.assertEqual(response.get_json(), test_data)

temp_list_labels = list(json_exe_dict.keys())


temp_list_test = ['1']

test = TestAPI()

start_time = datetime.now()

if test_decim:

    test_result = {'filename':[],'y_true':[], 'y_predict':[], 'proc_time':[]}
    
for label in temp_list_labels: # json_exe_dict

    for i in json_exe_dict[label][9:49]:
        

        if_pass = 1
        counter = 0
        while (if_pass and counter < 5):
            try:
                if server_mode == 'link':
                    test_data = {"test_file_name": f"BVH_EXPORT/{i}.json", 
                                 'mode':server_mode,
                                 'multi_cpu':multi_cpu}
                elif server_mode == 'data':
                    with open(f"BVH_EXPORT/{i}.json", 'rb') as json_data:
                        data = json.load(json_data)
                        json_data.close()
                    test_data = {"test_file_name": f"BVH_EXPORT/{i}.json", 
                                 'mode':server_mode, 
                                 'data':data,
                                 'multi_cpu':multi_cpu}
                else:
                    print('Server mode did not defined: Error!')

                
                result = test.test_process_json(server_url, test_data)

                if_pass = 0
            except:
                print(f'Error in record{i} and test label {label} in {counter} attempt.')
                counter += 1
                pass
            if counter >=5:
                raise Exception("We are reached maximum number of attempts")

        test_result['filename'].append(i)
        test_result['y_true'].append(label)
        test_result['y_predict'].append(result['label'])
        test_result['proc_time'].append(round(result['Processing time'], 1))

       
        
test_result_df = pd.DataFrame(test_result)

score_all = precision_recall_fscore_support(test_result_df['y_true'],
                                            test_result_df['y_predict'], 
                                            average=None, 
                                            labels=temp_list_labels)

score_all_df = pd.DataFrame(score_all, 
                            columns=temp_list_labels, 
                            index=['precision','recall','fscore','records'])
   

time_stat = []

for i in temp_list_labels:
    temp_df = test_result_df[test_result_df['y_true']==i]
    time_stat.append({'label':i, 
                      'min_time':round(temp_df['proc_time'].min(),1), 
                      'max_time':round(temp_df['proc_time'].max(),1),
                      'median_time':round(temp_df['proc_time'].median(),1)})

time_stat_df = pd.DataFrame(time_stat).set_index('label', inplace=False)

score_all_df = score_all_df.T

score_all_df['precision'] = score_all_df['precision'].round(decimals=2)

score_all_df['recall'] = score_all_df['recall'].round(decimals=2)

score_all_df['fscore'] = score_all_df['fscore'].round(decimals=2)

delta = datetime.now() - start_time

print(f'Processing time: {round(delta.total_seconds()/60)}, min.')
    
#-----------------------------------------------------------------------------#
    
