import time
import tensorflow as tf
import os
import numpy as np
import json

with open('herolist.json', 'r', encoding='utf-8') as f:
    hero_data = json.load(f)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
dim = 126

saved_model = tf.keras.models.load_model('saved_model/20250501_20250601.h5')

heroScoreList = []
heroTimesList = []
testPoolList = []
hero_names = []
for i in range(0, dim):
    hero_names.append(hero_data[str(i + 1)])
    heroScoreList.append(0)
    heroTimesList.append(0)

for i in range(1, 1000):
    totalCount = 0
    newArray = []

    while totalCount < 32:
        rnd = np.random.randint(0, dim + 1)
        if rnd not in newArray:
            newArray.append(rnd)
            totalCount = totalCount + 1

    testPoolList.append(newArray)


def takeSecond(elem):
    return elem[1]


for hero_pool in testPoolList:
    test_pool = []
    hero_id_list = []
    for hero_index, hero_name in hero_data.items():
        if int(hero_index) in hero_pool:
            hero_id_list.append(hero_index)
            test_pool.append(1)
        else:
            test_pool.append(0)

    test_in = []
    for hero_index, is_exist in enumerate(test_pool):
        if is_exist == 1:
            hero_select = np.zeros(dim)
            hero_select[hero_index] = 1
            test_in.append([test_pool, hero_select])

    test_in = np.array(test_in)
    out_list = saved_model(test_in, training=False)

    result_list = []
    for idx, item in enumerate(out_list):
        result_list.append([hero_id_list[idx], round(float(item[0]), 3)])

    result_list.sort(key=takeSecond, reverse=True)

    for item in result_list:
        heroIndex = int(item[0]) - 1
        heroScoreList[heroIndex] = heroScoreList[heroIndex] + item[1]
        heroTimesList[heroIndex] = heroTimesList[heroIndex] + 1

for i in range(0, dim):
    if heroTimesList[i] > 0:
        print(hero_names[i], "\t", round(float(heroScoreList[i] / heroTimesList[i]), 3))
    else:
        print(hero_names[i], "\t", 0)

