'''
Adapted from Lucid official tutorial notebook
https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/tutorial.ipynb

'''
#%%
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform

# Lucid's modelzoo can be accessed as classes in vision_models
import lucid.modelzoo.vision_models as models
# ... orA throguh a more systematic factory API
import lucid.modelzoo.nets_factory as nets
model = models.AlexNet()
model.load_graphdef()
# model = models.InceptionV1()
# model.load_graphdef()
#
# _ = render.render_vis(model, "mixed4a_pre_relu:476")
#
# obj = objectives.channel("mixed4a_pre_relu", 465)
# _ = render.render_vis(model, obj)

#%%
vec = np.random.randn(384).astype(np.float32)
obj = objectives.direction_cossim("conv4_1", vec=vec)
param_f = lambda: param.image(128, fft=True, decorrelate=True)
img = render.render_vis(model, obj, param_f)
plt.imsave("vis.png", img[0][0,:])

#%%
