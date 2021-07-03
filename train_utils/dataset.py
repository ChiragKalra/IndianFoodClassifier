from tensorflow.keras.preprocessing import image_dataset_from_directory


SEED = 1214

all_dataset = 'datasets\\all_classes'
fruit_dataset = 'D:\\PyCharm Projects\\IndianFoodClassfier\\fruits-360\\Training'

fruit360_classes = []
all_class_names = []


def food_dataset(batch_size, img_size):
	train_dataset = image_dataset_from_directory(
		all_dataset,
		batch_size=batch_size,
		shuffle=True,
		image_size=img_size,
		label_mode='categorical',
		validation_split=0.01,
		subset='training',
		seed=SEED,
	)
	global all_class_names, fruit360_classes
	all_class_names = train_dataset.class_names

	cv_dataset = image_dataset_from_directory(
		all_dataset,
		batch_size=batch_size,
		shuffle=True,
		image_size=img_size,
		label_mode='categorical',
		validation_split=0.01,
		subset='validation',
		seed=SEED,
	)

	dataset = image_dataset_from_directory(
		fruit_dataset,
		batch_size=batch_size,
		image_size=img_size,
		label_mode='categorical',
		seed=SEED,
	)
	fruit360_classes = dataset.class_names

	return train_dataset, cv_dataset
