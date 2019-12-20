
from load_neural_data import ExpTable, ExpData

#%%
ftr = (ExpTable.Expi == 12) & ExpTable.expControlFN.str.contains("generate")
print(ExpTable.comments[ftr].str.cat())
EData = ExpData(ExpTable[ftr].ephysFN.str.cat(), ExpTable[ftr].stimuli.str.cat())
EData.load_mat()
#%%
ftr = (ExpTable.Expi == 12) & ExpTable.expControlFN.str.contains("selectivity")
print(ExpTable.comments[ftr].str.cat())
MData = ExpData(ExpTable[ftr].ephysFN.str.cat(), ExpTable[ftr].stimuli.str.cat())
MData.load_mat()
#%%
pref_ch_idx = (MData.spikeID == 20).nonzero()[1]


#%%
import tensorflow as tf
from alexnet.alexnet import MyAlexnet
net = MyAlexnet()
#%%
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
init = tf.initialize_all_variables()
sess = tf.Session(config=config)
sess.run(init)

t = time.time()
output = sess.run(net.conv4, feed_dict = {net.x:[im1,im2]})
#%%
MData.close()
EData.close()

#%%
stimnames = [join(MData.stimuli, imgfn)+".jpg" for imgfn in MData.imgnms]
imgpaths_ds = tf.data.Dataset.from_tensor_slices(stimnames)
def load_preprocess(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0  # normalize to [0,1] range
    return image
image_ds = imgpaths_ds.map(load_preprocess)
#%% tensorflow input pipeline
img_raw = tf.io.read_file(stimnames[0])
img_tensor = tf.image.decode_image(img_raw)
print(img_tensor.shape)
print(img_tensor.dtype)
#%%

#%%
stimnames = [join(MData.stimuli, imgfn)+".jpg" if "gab_ori_" not in imgfn else join(MData.stimuli, imgfn)+".bmp"
             for imgfn in MData.imgnms]
imgpaths_ds = tf.data.Dataset.from_tensor_slices(stimnames[:30])
def load_preprocess(path):
    image = tf.io.read_file(path)
    image = tf.cond(
        tf.image.is_jpeg(image),
        lambda: tf.image.decode_jpeg(image, channels=3),
        lambda: tf.image.grayscale_to_rgb(tf.image.decode_bmp(image, channels=1)))
    image = tf.image.resize(image, [224, 224])
    image /= 255.0  # normalize to [0,1] range
    return image

image_ds = imgpaths_ds.map(load_preprocess)
Batchs = image_ds.batch(10)
iterator = Batchs.make_initializable_iterator()#Batchs.make_one_shot_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    #ids, img_batch = next(iter(Batchs))
    sess.run(iterator.initializer)
    while True:
        try:
            # value = sess.run(next_element)
            output = sess.run(net.conv4, feed_dict={net.x: next_element})
            print(output)
        except tf.errors.OutOfRangeError:
            break
    # imgs = sess.run(next_element)

    # for ids, img_batch in Batchs:
    #     output = sess.run(net.conv4, feed_dict = {net.x:img_batch})
    #     print(output)
