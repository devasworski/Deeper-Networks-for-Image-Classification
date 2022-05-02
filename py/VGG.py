from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import Model

def vgg16(classes=10, name="VGG_16"):
	"""
	> The function takes in the number of classes and the name of the model as input and returns a Keras
	model with the architecture of VGG16
	
	:param classes: number of classes in the dataset, defaults to 10 (optional)
	:param name: The name of the model, defaults to VGG_16 (optional)
	:return: A model object
	"""

	#input
	input_layer = Input(shape = (224,224,3))

	# block 1
	x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu',  padding='same')(input_layer)
	x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

	# block 2
	x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

	# block 3
	x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

	# block 4
	x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

	# block 5
	x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

	x = Flatten()(x)
	x = Dense(units=4096, activation='relu')(x)
	x = Dropout(0.5, name='dropout')(x)
	x = Dense(classes, activation='softmax')(x)	
	
	return Model(input_layer, x, name=name)