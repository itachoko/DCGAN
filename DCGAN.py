import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import loadmat

#トレーニング用
trainset=loadmat("data/train_32x32.mat")
#Xが画像データ,Yが正解データ,73257個のデータ
testset=loadmat("data/test_32x32.mat")

#データのスケールを0~255-1~1にする
def scale(x,feature_ranges=(-1,1)):
    x=((x-x.min())/(255-x.min()))
    min,max=feature_ranges
    x=x*(max-min)+min
    return x

#Datasetのクラスを定義する(ミニバッチを生成するとかデータセットを扱うクラス)
#validation=学習
class Dataset:
    #val_fracはtestデータを学習中用と学習後用に分離する
    def __init__(self,train,test,val_frac=0.5,shuffle=False,scale_func=None):
        split_index=int(len(test["y"])*(1-val_frac))
        #{r,g,b,index}
        self.test_x,self.valid_x=test["X"][:,:,:,:split_index],test["X"][:,:,:,split_index:]
        self.test_y,self.valid_y=test["y"][:split_index],test["y"][split_index:]
        self.train_x,self.train_y=train["X"],train["y"]
        #matlab形式{R.G,B,index}からtensorflow形式{index,r,g,b}
        #先頭から３番目を一番前に持っていく
        self.train_x=np.rollaxis(self.train_x,3)
        self.valid_x=np.rollaxis(self.valid_x,3)
        self.test_x=np.rollaxis(self.test_x,3)
        #スケール関数
        if scale_func is None:
            self.scaler=scale
        else:
            self.scaler=scaler_func
        self.shuffle=shuffle
        
  #ミニバッチ生成の関数
    def batches(self,batch_size):
        #self.shaffleでシャッフルするか判断
        if self.shuffle:
            #シャッフルする
            idx=np.arange(len(dataset.train_x))
            np.random.shuffle(idx)
            self.train_x=self.train_x[idx]
            self.train_y=self.train_y[idx]

        n_batches=len(self.train_y)//batch_size
        #range([始まりの数値,] 最後の数値[, 増加する量])
        for ii in range(0,len(self.train_y),batch_size):
            x=self.train_x[ii:ii+batch_size]
            y=self.train_y[ii:ii+batch_size]
            #scalerで-1~1の範囲に
            yield self.scaler(x),y
        
#プレースホルダーを初期化する関数
def model_inputs(real_dim,z_dim):
    #*で可変長
    inputs_real=tf.placeholder(tf.float32,(None,*real_dim),name="input_real")
    inputs_z=tf.placeholder(tf.float32,(None,z_dim),name="input_z")
    
    return inputs_real,inputs_z

#ジェネレータの定義
def generator(z,output_dim,reuse=False,alpha=0.2,training=True):
    with tf.variable_scope("generator",reuse=reuse):
        x1=tf.layers.dense(z,4*4*512)
        #3次元データを1次元データに
        x1=tf.reshape(x1,(-1,4,4,512))
        #バッチノーマライゼーション
        x1=tf.layers.batch_normalization(x1,training=training)
        #Leaky Relu
        x1=tf.maximum(alpha*x1,x1)
        
        #畳み込み
        #出力を256,フィルターのサイズ5,ストライド2,入力値と出力値がおなじになるように
        #↓Deconvolution 畳み込みの逆
        x2=tf.layers.conv2d_transpose(x1,256,5,strides=2,padding="same")
        x2=tf.layers.batch_normalization(x2,training=training)
        x2=tf.maximum(alpha*x2,x2)
        #8*8*256になる
        
        x3=tf.layers.conv2d_transpose(x2,128,5,strides=2,padding="same")
        x3=tf.layers.batch_normalization(x3,training=training)
        x3=tf.maximum(alpha*x3,x3)
        #16*16*128になる
        
        logits=tf.layers.conv2d_transpose(x3,output_dim,5,strides=2,padding="same")
        #32*32*3
        
        #活性化関数
        out=tf.tanh(logits)
        
        return out


def discriminator(x,reuse=False,alpha=0.2):
    with tf.variable_scope("discriminator",reuse=reuse):
        x1=tf.layers.conv2d(x,64,5,strides=2,padding="same")
        x1=tf.maximum(alpha*x1,x1)
        #16*16*64になる
        
        x2=tf.layers.conv2d(x1,128,5,strides=2,padding="same")
        x2=tf.layers.batch_normalization(x2,training=True)
        x2=tf.maximum(alpha*x2,x2)
        #8*8*128
        
        x3=tf.layers.conv2d(x2,256,5,strides=2,padding="same")
        x3=tf.layers.batch_normalization(x3,training=True)
        x3=tf.maximum(alpha*x3,x3)
        #4*4*256
        
        #一次元データに
        flat=tf.reshape(x3,(-1,4*4*256))
        logits=tf.layers.dense(flat,1)
        out=tf.sigmoid(logits)
        
        return out,logits
                

#損失関数の定義
def model_loss(input_real,input_z,output_dim,alpha=0.2):
    g_model=generator(input_z,output_dim,alpha=alpha)
    d_model_real,d_logits_real=discriminator(input_real,alpha=alpha)
    d_model_fake,d_logits_fake=discriminator(g_model,reuse=True,alpha=alpha)
    
    #tf.reduce_mean→平均
    d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                               (logits=d_logits_real
                              ,labels=tf.ones_like(d_model_real)))
                               
    d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                              (logits=d_logits_fake,
                               labels=tf.zeros_like(d_model_fake)))
                               
    g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                         (logits=d_logits_fake,
                         labels=tf.ones_like(d_model_fake)))
    
    d_loss=d_loss_real+d_loss_fake
    
    return d_loss,g_loss
    
#最適化関数を定義
#adam_optの減衰率beta1
def model_opt(d_loss,g_loss,learning_rate,bata1):
    t_vars=tf.trainable_variables()
    #d,gそれぞれのパラメータを
    d_vars=[var for var in t_vars if var.name.startswith("discriminator")]
    g_vars=[var for var in t_vars if var.name.startswith("generator")]
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt=tf.train.AdamOptimizer(learning_rate,beta1).minimize(d_loss,var_list=d_vars)
        g_train_opt=tf.train.AdamOptimizer(learning_rate,beta1).minimize(g_loss,var_list=g_vars)
        
    return d_train_opt,g_train_opt

#モデルのテンプレートを定義
class GAN:
    def __init__(self,real_size,z_size,learning_rate,alpha=0.2,beta1=0.5):
        tf.reset_default_graph()
        
        self.input_real,self.input_z=model_inputs(real_size,z_size)
        
        self.d_loss,self.g_loss=model_loss(self.input_real,self.input_z,real_size[2],alpha=alpha)
        
        self.d_opt,self.g_opt=model_opt(self.d_loss,self.g_loss,learning_rate,beta1)


 #生成した画像表示samples=画像データセット,ncols,nrows=何行何列で表示するか
def view_samples(epoch,samples,nrows,ncols,figsize=(5,5)):
    fig,axes=plt.subplots(figsize=figsize,nrows=nrows,ncols=ncols,sharey=True,sharex=True)
    for ax,img in zip(axes.flatten(),samples[epoch]):
        ax.axis("off")
        #astype(unsiged-int_adjustable)
        img=((img-img.min())*255/(img.max()-img.min())).astype(np.uint8)
        ax.set_adjustable('box')
        #box-forced四角形
        im=ax.imshow(img,aspect="equal")
        #縦横比一定
    plt.subplots_adjust(wspace=0,hspace=0)
             
    return fig,axes         



#トレーニングの関数の定義
#print_every途中結果を表示,show_every生成した画像を　何個表示するか

def train(net,dataset,batch_size,print_every=10,show_every=10,figsize=(5,5)):
    saver=tf.train.Saver()
    sample_z=np.random.uniform(-1,1,size=(72,z_size))
    
    samples,losses=[],[]
    steps=0
    
    with tf.Session() as sess:
        #変数のリセット
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for x,y in dataset.batches(batch_size):
                steps+=1
                
                batch_z=np.random.uniform(-1,1,size=(batch_size,z_size))
                
                _=sess.run(net.d_opt,feed_dict={net.input_real:x,net.input_z:batch_z})
                _=sess.run(net.g_opt,feed_dict={net.input_z:batch_z,net.input_real:x})
                
                #１０回毎に結果を表示
                if steps%print_every==0:
                    train_loss_d =net.d_loss.eval({net.input_z:batch_z,net.input_real:x})
                    train_loss_g=net.g_loss.eval({net.input_z:batch_z})
                    
                    print("Epoch{}/{}:".format(e+1,epochs),
                         "D Loss:{:.4f}".format(train_loss_d),
                         "G Loss:{:.4f}".format(train_loss_g))
                    
                    losses.append((train_loss_d,train_loss_g))
                
                if steps % show_every==0:
                    gen_samples=sess.run(generator(net.input_z,3,reuse=True,training=False),
                                        feed_dict={net.input_z:sample_z})
                    samples.append(gen_samples)
                    _=view_samples(-1,samples,6,12,figsize=figsize)
                    plt.show()
                    
       # 学習全て終わったら書き込みや            
        saver.save(sess,"./checkpoints/generator.ckpt")
        
    with open("samples.pkl","wb") as f:
        pkl.dump(samples,f)
        
    return losses,samples


#32*32*(r,g,b)
real_size=(32,32,3)
z_size=100
#学習率
learning_rate=0.0002
batch_size=128
epochs=25
#dのすけーるへんこう
alpha=0.2
#adamopt減衰パラ
beta1=0.5

net=GAN(real_size,z_size,learning_rate,alpha=alpha,beta1=beta1)

dataset=Dataset(trainset,testset)

losses,samples=train(net,dataset,epochs,batch_size,figsize=(10,5))