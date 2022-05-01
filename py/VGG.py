from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.layers.merge import concatenate
from keras import Model

def vgg16(classes=10, name="VGG_16"):

	#input
	input_layer = Input(shape = (224,224,3))

	# block 1
	x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',  padding='same')(input_layer)
	x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

	# block 2
	x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

	# block 3
	x = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

	# block 4
	x = layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

	# block 5
	x = layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
	x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

	x = layers.Flatten()(x)
	x = layers.Dense(units=4096, activation='relu')(x)
	x = layers.Dense(units=4096, activation='relu')(x)
	x = layers.Dense(units=1000, activation='relu')(x)
	x = layers.Dropout(0.5, name='dropout')(x)
	x = layers.Dense(classes, activation='softmax')(x)	
	
	return Model(input_layer, x, name=name)