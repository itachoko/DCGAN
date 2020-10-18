import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import loadmat

#トレーニング用
trainset=loadmat("data/train_32x32.mat")
#Xが画像データ,Yが正解データ,73257個のデータ
testset=loadmat("data/test_32x32.mat")

#ランダムに表示
#0~73257の値を36個生成
idx=np.random.randint(0,trainset["X"].shape[3],size=36)

print(trainset["X"].shape)
print(trainset["X"][:,:,:,0])