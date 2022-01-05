# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 00:20:06 2022

@author: User
"""
import os
import numpy as np
import torch
import pandas as pd
import pickle
import torchvision.ops as ops
root = "C:/Users/User/OneDrive/桌面/NYCU/final_project"
os.chdir(root)


f = open('C:/Users/User/OneDrive/桌面/mmdetection-master/result_retinanet/result.pkl', 'rb')
data = pickle.load(f)

# print(data)
answer = dict.fromkeys(range(0,3000))
resize_ratio = 1
for i, j in enumerate(data):
    # output = j[0][0]
    output = j[0]
    s = np.where(output[:,4] > 0.1)[0]
    result = ops.nms(torch.tensor(output[s,0:4]), torch.tensor(output[s,4]), 0.1)
    for k in result:
        # print(k)
        x = str(output[k,0] + 0.5 * (1-resize_ratio) * (output[k,2] - output[k,0]))
        y = str(output[k,1] + 0.5 * (1-resize_ratio) * (output[k,3] - output[k,1]))
        w = str(resize_ratio * (output[k,2] - output[k,0]))
        h = str(resize_ratio * (output[k,3] - output[k,1]))
        s = output[k,4]

        if answer[i] is None:
        # if s > 0.98 and test_df['PredictionString'][i] == '0.5 0 0 100 100':
            answer[i] = str(s) + " " + x + " " + y + " " + w + " " + h + " "
            # test_df['PredictionString'][i] = str(s) + " " + x + " " + y + " " + w + " " + h + " "
        else:
            answer[i] += str(s) + " " + x + " " + y + " " + w + " " + h + " "
            # test_df['PredictionString'][i] += str(s) + " " + x + " " + y + " " + w + " " + h + " "
        

print(len(answer))

test_df = pd.read_csv("stage_2_sample_submission.csv")
test_df['PredictionString'] = answer.values()
test_df.to_csv('answer.csv',index=False)


# test_df['PredictionString'] = None
# test_df2 = pd.read_csv("model1_test.csv")
# test_df2['PredictionString'] = answer.values()

# out = test_df.merge(test_df2, left_on=['patientId','PredictionString'], right_on=['patientId','PredictionString'], how='left')
# out = test_df.merge(test_df2, left_on=['patientId'], right_on=['patientId'], how='left')
# out = out.drop(columns=['PredictionString_x', 'pred'], axis = 1)
# out = out.rename(columns = {'PredictionString_y': 'PredictionString'})
# out.to_csv('answer.csv',index=False)