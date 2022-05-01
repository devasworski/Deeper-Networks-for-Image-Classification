# For the model definition, the code from [Khuyen Le](https://medium.com/mlearning-ai/implementation-of-googlenet-on-keras-d9873aeed83c) was used as a basis and then adjusted using the Guide from [Jai Faizy ](https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/) and the architecture of the Model was then updated to [V2](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202) 
from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.layers.merge import concatenate
from keras import Model

def inception(x, filter_1_conv1x1, filter_1_conv1x1_3x3, filter_2_conv1x1_3x3, filter_1_conv1x1_5x5, filter_2_conv1x1_5x5, filter_2_conv3x3_1x1):
	# 1x1
	conv1x1 = Conv2D(filters=filter_1_conv1x1, kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)

	# 1x1->3x3
	conv1x1_3x3 = Conv2D(filters=filter_1_conv1x1_3x3, kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
	conv1x1_3x3 = Conv2D(filters=filter_2_conv1x1_3x3, kernel_size=(3,3), strides=1, padding='same', activation='relu')(conv1x1_3x3)
	
	# 1x1->3x3->3x3
	conv1x1_5x5 = Conv2D(filters=filter_1_conv1x1_5x5, kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
	conv1x1_5x5 = Conv2D(filters=filter_2_conv1x1_3x3, kernel_size=(3,3), strides=1, padding='same', activation='relu')(conv1x1_5x5)
	conv1x1_5x5 = Conv2D(filters=filter_2_conv1x1_3x3, kernel_size=(3,3), strides=1, padding='same', activation='relu')(conv1x1_5x5)

	# 3x3->1x1
	conv3x3_1x1 = MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(x)
	conv3x3_1x1 = Conv2D(filters=filter_2_conv3x3_1x1, kernel_size=(1,1), strides=1, padding='same', activation='relu')(conv3x3_1x1)

	return concatenate([conv1x1, conv1x1_3x3, conv1x1_5x5, conv3x3_1x1], axis = -1)


def auxiliary(x, num_classes,name=None):

	x = layers.AveragePooling2D(pool_size=(5,5), strides=3,  padding='valid')(x)
	x = layers.Conv2D(filters=128,  kernel_size=(1,1),  strides=1,  padding='same', activation='relu')(x)
	x = layers.Flatten()(x)
	x = layers.Dense(units=256, activation='relu')(x)
	x = layers.Dropout(0.4)(x)
	x = layers.Dense(units=num_classes, activation='softmax', name=name)(x)
	return x

def googlenet(num_classes, name="GoogLeNet"):

	#input
	input_layer = Input(shape = (224, 224, 3))

	# Stage 1
	x = Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same', activation='relu')(input_layer)
	x = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x)
	x = BatchNormalization()(x)

	# Stage 2
	x = Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
	x = Conv2D(filters=192, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x)

	# Stage 3
	x = inception(x, 64,  96, 128, 16, 32, 32)
	x = inception(x, 128, 128, 192, 32, 96, 64)
	x = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x)
	
	# Stage 4
	x = inception(x, 192,  96, 208,  16, 48,  64)
	aux1 = auxiliary(x, num_classes, name='aux1')
	x = inception(x, 160, 112,224, 24,64,  64)
	x = inception(x, 128, 128,256, 24,64,  64)
	x = inception(x, 112, 144,288, 32,64,  64)
	aux2 = auxiliary(x, num_classes, name='aux2')
	x = inception(x, 256, 160, 320, 32, 128, 128)
	x = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x)
	
	# Stage 5
	x = inception(x, 256, 160, 320, 32, 128, 128)
	x = inception(x, 384, 192, 384, 48, 128, 128)
	x = GlobalAveragePooling2D()(x)
	
	# Output (Stage 6)
	x = Dropout(0.4)(x)
	x = Dense(num_classes, activation = 'softmax')(x)
  
	return  Model(input_layer, [x, aux1, aux2], name = name)