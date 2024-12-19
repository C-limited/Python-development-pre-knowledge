import os
import shutil
import tqdm
import pandas as pd
from PIL import Image
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
import csv


def create_directory_structure(base_path, sub_dirs):
    for sub_dir in sub_dirs:
        path = os.path.join(base_path, sub_dir)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f'Created directory: {path}')


def save_images_to_folders(data_csv_path, images_path, create_image_path):
    train_data = pd.read_csv(data_csv_path)
    create_directory_structure(create_image_path, ['train_images', 'test_images'])

    for label in train_data.columns[1:]:
        label_dir = os.path.join(create_image_path, 'train_images', label)
        create_directory_structure(create_image_path, [f'train_images/{label}'])

        for image_id, label_value in zip(train_data['image_id'], train_data[label]):
            if label_value == 1:
                img_name = f'{image_id}.jpg'
                img = Image.open(os.path.join(images_path, img_name))
                img.save(os.path.join(label_dir, img_name))

    for img_name in os.listdir(images_path):
        if img_name.startswith('Test'):
            img = Image.open(os.path.join(images_path, img_name))
            img.save(os.path.join(create_image_path, 'test_images', img_name))


def split_train_val(img_dir, val_num=0.2):
    create_val_image_path = './data_new/val_images'
    create_directory_structure(create_val_image_path, os.listdir(img_dir))

    for label in os.listdir(img_dir):
        label_dir = os.path.join(img_dir, label)
        images = os.listdir(label_dir)
        val_size = int(len(images) * val_num)

        for img in tqdm.tqdm(images[:val_size]):
            shutil.move(os.path.join(label_dir, img), os.path.join(create_val_image_path, label, img))


def build_model(input_shape, class_num, activation):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(class_num, activation=activation)
    ])
    return model


def plot_metrics(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'o', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title("Training and Validation accuracy")
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'o', label='Training accuracy')
    plt.plot(epochs, val_loss, 'r', label='Validation accuracy')
    plt.title("Training and Validation loss")
    plt.legend()
    plt.show()


def main():
    data_csv_path = './train.csv'
    images_path = './images'
    create_image_path = './data_new'
    val_num = 0.2
    model_name = 'model_224_150_1.h5'
    class_num = 4
    input_shape = (224, 224, 3)
    activation = 'softmax'
    epoch = 600
    batch = 20

    # Prepare data
    save_images_to_folders(data_csv_path, images_path, create_image_path)
    split_train_val(os.path.join(create_image_path, 'train_images'), val_num)

    train_dir = os.path.join(create_image_path, 'train_images')
    validation_dir = os.path.join(create_image_path, 'val_images')

    train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=40, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=input_shape[:2], batch_size=batch,
                                                        class_mode='sparse')
    validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=input_shape[:2],
                                                            batch_size=batch,
                                                            class_mode='sparse')

    model = build_model(input_shape, class_num, activation)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['acc'])

    history = model.fit_generator(train_generator, steps_per_epoch=len(train_generator), epochs=epoch,
                                  validation_data=validation_generator, validation_steps=len(validation_generator))

    model.save(os.path.join('./models', model_name))

    plot_metrics(history)

    # Inference and save results
    img_path = 'data_new/test_images/'
    img_list = os.listdir(img_path)
    class_list = ['healthy', 'multiple_diseases', 'rust', 'scab']
    model = tf.keras.models.load_model(os.path.join('./models', model_name))

    with open('predict_result.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab'])

        for img_name in img_list:
            img_full_path = os.path.join(img_path, img_name)
            img = image.load_img(img_full_path, target_size=input_shape[:2])
            img_tensor = image.img_to_array(img) / 255.
            img_tensor = np.expand_dims(img_tensor, axis=0)

            predictions = model.predict(img_tensor)[0]
            writer.writerow([img_name[:-4]] + predictions.tolist())

            predicted_class = class_list[np.argmax(predictions)]
            plt.imshow(image.array_to_img(img_tensor[0]))
            plt.title(predicted_class)
            plt.show()


if __name__ == '__main__':
    main()
