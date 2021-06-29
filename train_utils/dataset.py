from tensorflow.keras.preprocessing import image_dataset_from_directory


SEED = 21

datasets = [
	('datasets\\cropped\\continental', 0.01),
	('datasets\\cropped\\indian_0', 0.2),
	('datasets\\cropped\\indian_1', 0.2),
	('datasets\\fruits-360\\Training', 0.02),
]

fruit360_classes = None
all_class_names = None


def food_dataset(batch_size, img_size):
	train_dataset = image_dataset_from_directory(
		'datasets\\all_classes',
		batch_size=batch_size,
		shuffle=True,
		image_size=img_size,
		label_mode='categorical',
		validation_split=0.5,
		subset='training',
		seed=SEED,
	)
	global all_class_names, fruit360_classes
	all_class_names = train_dataset.class_names

	cv_dataset = image_dataset_from_directory(
		'datasets\\all_classes',
		batch_size=batch_size,
		shuffle=True,
		image_size=img_size,
		label_mode='categorical',
		validation_split=0.5,
		subset='validation',
		seed=SEED,
	)

	for directory, split in datasets:
		dataset = image_dataset_from_directory(
			directory,
			batch_size=batch_size,
			shuffle=True,
			image_size=img_size,
			label_mode='categorical',
			validation_split=split,
			subset='training',
			seed=SEED,
		)
		if 'fruits-360' in directory:
			fruit360_classes = dataset.class_names
		train_dataset = train_dataset.concatenate(dataset)

		dataset = image_dataset_from_directory(
			directory,
			batch_size=batch_size,
			shuffle=True,
			image_size=img_size,
			label_mode='categorical',
			validation_split=split,
			subset='validation',
			seed=SEED,
		)
		cv_dataset = cv_dataset.concatenate(dataset)
		print()

	return train_dataset, cv_dataset
