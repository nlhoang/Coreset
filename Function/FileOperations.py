import numpy as np
import csv


# --------------------------------------------------------------------------------------
# Function to read file
# Using for set_a, set_s, miscellaneous, unbalance
def OpenFile1(fileName):
    data = []
    file = open(fileName)
    text = file.read().splitlines()
    for temp in text:
        val = list(filter(None, temp.split(' ')))
        data += [[float(val[0]), float(val[1])]]
    file.close()
    return data


# Using for shape_sets with label
def OpenFile2(fileName):
    data = []
    label = []
    file = open(fileName)
    text = file.read().splitlines()
    for temp in text:
        val = list(filter(None, temp.split('\t')))
        data += [[float(val[0]), float(val[1])]]
        label += [int(val[2])]
    file.close()
    return data


# Using for KDD Cup, MINST
def OpenFile3(fileName, dim):
    data = []
    file = open(fileName)
    text = file.read().splitlines()
    for temp in text:
        row = []
        val = list(filter(None, temp.split(' ')))
        for i in range(dim):
            row += [float(val[i])]
        data += [row]
    file.close()
    return data


# Using for ConfLongDemo
def OpenFile4(fileName, dim):
    data = []
    file = open(fileName)
    text = file.read().splitlines()
    for temp in text:
        row = []
        val = list(filter(None, temp.split('\t')))
        for i in range(dim):
            row += [float(val[i])]
        data += [row]
    file.close()
    return data


# --------------------------------------------------------------------------------------
# Read data
# [0]link, [1]name, [2]size, [3]K-cluster, [4]dim, #[5]function, [6]code, [7]csmaxsize
def GetData(index):
    all_link = [
        ['data/shape_sets/flame.txt', 'Flame', 240, 2, 2, 2, 'D1', 200, 0],
        ['data/shape_sets/jain.txt', 'Jain', 373, 2, 2, 2, 'D2', 200, 1],
        ['data/shape_sets/Aggregation.txt', 'Aggregation', 788, 7, 2, 2, 'D3', 300, 2],
        ['data/shape_sets/R15.txt', 'R15', 600, 15, 2, 2, 'D4', 300, 3],
        ['data/shape_sets/D31.txt', 'D31', 3100, 31, 2, 2, 'D5', 1000, 4],
        ['data/unbalance/unbalance.txt', 'Unbalance', 6500, 8, 2, 1, 'D6', 2500, 5],
        ['data/set_a/a1.txt', 'A1', 3000, 20, 2, 1, 'D7', 1000, 6],
        ['data/set_a/a2.txt', 'A2', 5250, 35, 2, 1, 'D8', 1500, 7],
        ['data/set_a/a3.txt', 'A3', 7500, 50, 2, 1, 'D9', 2500, 8],
        ['data/set_s/s1.txt', 'S1', 5000, 15, 2, 1, 'D10', 1500, 9],
        ['data/set_s/s2.txt', 'S2', 5000, 15, 2, 1, 'D11', 1500, 10],
        ['data/set_s/s3.txt', 'S3', 5000, 15, 2, 1, 'D12', 1500, 11],
        ['data/set_s/s4.txt', 'S4', 5000, 15, 2, 1, 'D13', 1500, 12],
        ['data/miscellaneous/t48k.txt', 't48k', 8000, 6, 2, 1, 'D14', 2500, 13],
        ['data/birch/birch1.txt', 'Birch1', 100000, 100, 2, 1, 'D15', 3000, 14],
        ['data/birch/birch2.txt', 'Birch2', 100000, 100, 2, 1, 'D16', 3000, 15],
        ['data/birch/birch3.txt', 'Birch3', 100000, 100, 2, 1, 'D17', 3000, 16],
        ['data/miscellaneous/MINST.txt', 'MINST', 10000, 10, 748, 3, 'D18', 2500, 17],
        ['data/miscellaneous/ConfLongDemo.txt', 'ConfLongDemo', 164860, 11, 3, 4, 'D19', 5000, 18],
        ['data/kddcup/KDDCUP04Bio.txt', 'KDDBio', 145751, 200, 74, 3, 'D20', 5000, 19]
    ]

    link = all_link[index][0]
    dataname = all_link[index][1]
    datasize = all_link[index][2]
    dim = all_link[index][4]
    K = all_link[index][3]
    func = all_link[index][5]
    datacode = all_link[index][6]
    csmaxsize = all_link[index][7]
    if func == 1:
        data = np.asarray(OpenFile1(link))
    elif func == 2:
        data = np.asarray(OpenFile2(link))
    elif func == 3:
        data = np.asarray(OpenFile3(link, dim))
    else:
        data = np.asarray(OpenFile4(link, dim))

    return (data, dataname, datasize, K, datacode, csmaxsize)


# --------------------------------------------------------------------------------------
# Read Results of ProTraS
def ReadProTraS(index):
    link = 'results/ProTraS_records/D' + str(index + 1) + '.csv'
    with open(link) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataname, datasize, coresetsize = row
                line_count += 1
                continue
            if line_count == 1:
                coreset = row
                line_count += 1
                continue
            if line_count == 2:
                timerec = row
                line_count += 1
                continue
            if line_count == 3:
                cluster_id = row
                line_count += 1
                continue
            if line_count == 4:
                cluster_dis = row

        datasize = int(datasize)
        coresetsize = int(coresetsize)
        coreset = np.asarray(coreset, dtype=int)
        timerec = np.asarray(timerec, dtype=float)
        cluster_id = np.asarray(cluster_id, dtype=float)
        cluster_id = cluster_id.astype(int)
        cluster_dis = np.asarray(cluster_dis, dtype=float)

    return coreset, timerec, cluster_id, cluster_dis
