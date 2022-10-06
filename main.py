import time
import tensorflow as tf
import os
import numpy as np
import json

with open('herolist.json', 'r', encoding='utf-8') as f:
    hero_data = json.load(f)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
dim = 123

saved_model = tf.keras.models.load_model('saved_model/2.h5')

testPool = [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0
            ]

print("英雄池：")
for hero_index, is_exist in enumerate(testPool):
    if is_exist == 1:
        print(hero_data[str(hero_index + 1)], end=",")
print("\n")

test_in = []
hero_names = []
for hero_index, is_exist in enumerate(testPool):
    if is_exist == 1:
        hero_select = np.zeros(dim)
        hero_select[hero_index] = 1
        hero_names.append(hero_data[str(hero_index + 1)])
        test_in.append([testPool, hero_select])

test_in = np.array(test_in)

clock = time.time()
# out_list = saved_model.predict(test_in)
out_list = saved_model(test_in, training=False)

print(time.time() - clock, 's')
result_list = []
for hero_index, item in enumerate(out_list):
    result_list.append([hero_names[hero_index], float(item[0])])


def takeSecond(elem):
    return elem[1]


result_list.sort(key=takeSecond, reverse=True)

print("推荐选择：")
recommend_count = 0
for item in result_list:
    print(item)
    recommend_count = recommend_count + 1
    if recommend_count > 20:
        break
