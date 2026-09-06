import numpy as np

from .config import (
    AUGMENTATION_SETTINGS,
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_SIZE,
    SEED,
)

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)
IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


def dense_preprocess(x):
    return x.astype(np.float32) / 255.0


def efficientnet_preprocess(x):
    return x.astype(np.float32) / 127.5 - 1.0


def convnext_preprocess(x):
    x = x.astype(np.float32) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


PREPROCESS_FUNCTIONS = {
    "efficientnet": efficientnet_preprocess,
    "densenet": dense_preprocess,
    "convnext": convnext_preprocess,
}


def make_generator(
    dataframe,
    model_name,
    training,
    batch_size=BATCH_SIZE,
    interpolation="nearest",
):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    preprocessing_function = PREPROCESS_FUNCTIONS[model_name]

    if training:
        datagen = ImageDataGenerator(
            preprocessing_function=preprocessing_function,
            **AUGMENTATION_SETTINGS,
        )
    else:
        datagen = ImageDataGenerator(
            preprocessing_function=preprocessing_function,
        )

    return datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col="path",
        y_col="class_name",
        classes=list(CLASS_NAMES),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=training,
        seed=SEED,
        validate_filenames=True,
        interpolation=interpolation,
    )
