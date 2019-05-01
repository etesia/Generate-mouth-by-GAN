import datetime
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0' 
from scipy.misc.pilutil import imread, imresize
from keras.models import *
from keras.layers import *
from keras.layers.merge import *
from keras.optimizers import *
import cv2

###############################################################
# Training Data List Creat
###############################################################
train_real_data_dir = r'.\Training\Real\*'
train_white_data_dir = r'.\Training\White\*'


real_list = glob.glob(train_real_data_dir)
train_real_data_list = []
train_real_data_list.extend(real_list)

white_list = glob.glob(train_white_data_dir)
train_white_data_list = []
train_white_data_list.extend(white_list)


df = 64 # num of d's filters
gf = 64 # num of g's filters

# ###############################################################
# # Define D and G and parameter
# ###############################################################
img_row = img_col = 128 
channels = 1    
img_shape=(img_row, img_col, channels) 

def dis(input_shape):
    def conv_block(layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d
     
    img_A = Input(input_shape)      
    img_B = Input(input_shape)      
    combined_imgs = Concatenate(axis=-1)([img_A,img_B])
    d1 = conv_block(combined_imgs, df, bn=False)
    d2 = conv_block(d1, 2*df)
    d3 = conv_block(d2, 4*df)
    d4 = conv_block(d3, 8*df)
   
    x = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
    
    model = Model([img_A,img_B], x)
    print('Model_Discriminator:')
    # model.summary()
    return model
    
    
def gen(input_shape):
    
    def conv_block(layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    
    def deconv_block(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)

        u = Concatenate()([u, skip_input]) # U net 架構
        return u

    d0 = Input(input_shape)   # d0 = imgA


    d1 = conv_block(d0, gf, bn=False) 
    d2 = conv_block(d1, 2*gf) 
    d3 = conv_block(d2, 4*gf) 
    d4 = conv_block(d3, 8*gf) 
    d5 = conv_block(d4, 8*gf) 
    d6 = conv_block(d5, 8*gf) 
    d7 = conv_block(d6, 8*gf)

    u1 = deconv_block(d7, d6, gf*8)
    u2 = deconv_block(u1, d5, gf*8)
    u3 = deconv_block(u2, d4, gf*8)
    u4 = deconv_block(u3, d3, gf*4)
    u5 = deconv_block(u4, d2, gf*2)
    u6 = deconv_block(u5, d1, gf)

    u7 = UpSampling2D(size = 2)(u6)
    
    out_img = Conv2D(channels, kernel_size=4, strides=1, padding='same',activation='tanh')(u7) 

    model = Model(d0, out_img)
    # print('Model_Generator:')
    # model.summary()
    return model


input_shape=(img_row, img_col, channels) # or (img_row,img_col,channels)
crop_shape=(img_row,img_col)
G = gen(input_shape)
D = dis(input_shape)


D_optimizer = Adam(0.0002, 0.5)
D.compile(loss='mse', optimizer=D_optimizer,metrics=['accuracy'])
# D.summary()

 
AM_optimizer = Adam(0.0002, 0.5)
img_A = Input(input_shape)          
img_B = Input(input_shape)         
fake_A = G(img_B)                   
D.trainable=False                   
valid = D([fake_A,img_B])           
AM = Model([img_A,img_B],[valid,fake_A])  

                          
AM.compile(loss=['mse', 'mae'],loss_weights=[1,50],optimizer=AM_optimizer)
# AM.summary()


def generator_training_Img(real_list_dir,white_list_dir,resize=None,batch_size=32):
    batch_real_img=[]
    batch_white_img=[]
    for _ in range(batch_size):
        random_index = int(np.random.randint(len(real_list_dir),size=1))
        real_img = imread(real_list_dir[random_index],mode='L')
        white_img = imread(white_list_dir[random_index],mode='L')
        if resize:
            real_img = imresize(real_img,resize)
            white_img = imresize(white_img,resize)
        batch_real_img.append(real_img)
        batch_white_img.append(white_img)
    batch_real_img = np.array(batch_real_img)/127.5-1
    batch_real_img = np.expand_dims(batch_real_img,axis=1)
    batch_white_img = np.array(batch_white_img)/127.5-1
    batch_white_img = np.expand_dims(batch_white_img,axis=1)
    return batch_real_img,batch_white_img


batch_size = 32 
all_epoch = 7000
all_d_loss = np.zeros(all_epoch)
all_g_loss = np.zeros(all_epoch)


valid = np.ones((batch_size, 8, 8,1))
fake  = np.zeros((batch_size, 8, 8, 1))
    

start_time=datetime.datetime.now()
for now_iter in range(all_epoch):
    ori_img,white_img = generator_training_Img(real_list_dir=train_real_data_list,
                                               white_list_dir=train_white_data_list,
                                               resize=(img_row,img_col),
                                               batch_size=batch_size)
    imgs_A = ori_img 
    imgs_B = white_img 
    imgs_B = imgs_B.reshape((32,128,128,1))
    imgs_A = imgs_A.reshape((32,128,128,1))
    
    
#     ###################################
#     #Training Discriminator Phase
#     ###################################
    fake_A = G.predict(imgs_B) 

    D_loss_Real = D.train_on_batch([imgs_A, imgs_B], valid)
    D_loss_Fake = D.train_on_batch([fake_A, imgs_B], fake)
    D_loss = 0.5 * np.add(D_loss_Real,D_loss_Fake)

    # G_loss = AM.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
    for i in range(4): # 這邊可選擇fine tune多次
        G_loss = AM.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
        
    all_d_loss[now_iter] = D_loss[0]
    all_g_loss[now_iter] = G_loss[0]

    end_time = datetime.datetime.now() - start_time
    print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss1: %f,loss2: %f] [time:%s]" % (now_iter,all_epoch,D_loss[0],D_loss[1]*100,G_loss[0],G_loss[1],end_time))
    np.savetxt("all_d_loss.txt", all_d_loss, delimiter=",")
    np.savetxt("all_g_loss.txt", all_g_loss, delimiter=",")

    
def generator_test_Img(white_list_dir,resize=None ):
    batch_real_img=[]
    batch_white_img=[]
    for i in range(10):
        white_img =  imread(white_list_dir[i] , mode='L')

        if resize:
            white_img = imresize(white_img,resize)
        batch_white_img.append(white_img)
    batch_white_img = np.array(batch_white_img)/127.5-1
    batch_white_img = np.expand_dims(batch_white_img,axis=3)
    return batch_white_img


test_white_data_dir = r'testimg/*'
test_white_list = glob.glob(test_white_data_dir)
test_white_data_list = []
test_white_data_list.extend(test_white_list)
test_white_data_list = sorted(test_white_data_list)

test_white_data_list = generator_test_Img( white_list_dir=test_white_data_list, resize=(128,128))

fake_A = G.predict(test_white_data_list)
gen_imgs = fake_A
# gen_imgs = np.concatenate([fake_A])

gen_imgs = 0.5 * gen_imgs + 0.5
# for plotting


ids = 0
for img in gen_imgs:
    img = img.reshape((128, 128))
    cv2.imwrite('res_images/cv_saves/'+str(ids)+ ".jpg", img*255)
    plt.imsave("res_images/main_test_res_" + str(ids) + ".jpg", img, cmap="gray")
    ids += 1                  
plt.close()   
print("test_data generator predict over.")


def numpy_to_csv(input_image,image_number=10,save_csv_name='predict.csv'):
    save_image=np.zeros([int(input_image.size/image_number),image_number],dtype=np.float32)

    for image_index in range(image_number):
        save_image[:,image_index]=input_image[image_index,:,:].flatten()

    base_word='id'
    df = pd.DataFrame(save_image)
    index_col=[]
    for i in range(n):
        col_word=base_word+str(i)
        index_col.append(col_word)
    df.index.name='index'
    df.columns=index_col
    df.to_csv(save_csv_name)
    print("Okay! numpy_to_csv")

n=10
numpy_to_csv(input_image= gen_imgs,image_number=n,save_csv_name='Predict.csv')


# draw loss 
all_d_loss_txt = np.loadtxt("all_d_loss.txt")
all_g_loss_txt = np.loadtxt("all_g_loss.txt")

# print( all_d_loss_txt.shape, all_d_loss_txt.shape[0])
# print(all_g_loss_txt, all_g_loss_txt.shape, all_g_loss_txt.shape[0])

fig = plt.figure()
ax = plt.axes()
all_d_loss_x = np.linspace(0, 1, all_d_loss_txt.shape[0])
all_g_loss_x = np.linspace(0, 1, all_g_loss_txt.shape[0])

plt.plot(all_g_loss_x, all_g_loss_txt, '-r');  # dotted red, g_loss
# plt.plot(all_d_loss_x , all_d_loss_txt , '-g');  # dotted green, d_loss

plt.show()
