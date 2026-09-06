from .config import (
    CONVNEXT_FINETUNE_LR,
    CONVNEXT_LR,
    DENSENET_LR,
    EFFICIENTNET_LR,
    IMAGE_SIZE,
    NUM_CLASSES,
)

BACKBONE_DIMS = {
    "efficientnet": 1280,
    "densenet": 1664,
    "convnext": 768,
}

BACKBONE_DISPLAY = {
    "efficientnet": "EfficientNetV2-M",
    "densenet": "DenseNet169",
    "convnext": "ConvNeXt-Tiny",
}


def build_backbone(name):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    input_tensor = keras.Input(
        shape=(*IMAGE_SIZE, 3),
        name=f"{name}_input",
    )

    if name == "efficientnet":
        try:
            base = tf.keras.applications.EfficientNetV2M(
                weights="imagenet",
                include_top=False,
                include_preprocessing=False,
                input_shape=(*IMAGE_SIZE, 3),
            )
        except TypeError:
            base = tf.keras.applications.EfficientNetV2M(
                weights="imagenet",
                include_top=False,
                input_shape=(*IMAGE_SIZE, 3),
            )

        for layer in base.layers[:-100]:
            layer.trainable = False
        for layer in base.layers[-100:]:
            layer.trainable = True

        x = base(input_tensor)
        embedding = layers.GlobalAveragePooling2D(
            name="efficientnet_embedding"
        )(x)
        x = layers.Dense(
            512,
            kernel_regularizer=keras.regularizers.l2(1e-3),
        )(embedding)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.30)(x)
        x = layers.Dense(
            256,
            kernel_regularizer=keras.regularizers.l2(1e-3),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.20)(x)
        logits = layers.Dense(
            NUM_CLASSES,
            name="efficientnet_logits",
        )(x)
        output = layers.Softmax(
            name="efficientnet_probabilities"
        )(logits)

    elif name == "densenet":
        base = tf.keras.applications.DenseNet169(
            weights="imagenet",
            include_top=False,
            input_shape=(*IMAGE_SIZE, 3),
        )

        for layer in base.layers[:-30]:
            layer.trainable = False
        for layer in base.layers[-30:]:
            layer.trainable = True

        x = base(input_tensor)
        embedding = layers.GlobalAveragePooling2D(
            name="densenet_embedding"
        )(x)
        x = layers.Dense(
            512,
            kernel_regularizer=keras.regularizers.l2(1e-3),
        )(embedding)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.40)(x)
        x = layers.Dense(
            256,
            kernel_regularizer=keras.regularizers.l2(1e-3),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.30)(x)
        logits = layers.Dense(
            NUM_CLASSES,
            name="densenet_logits",
        )(x)
        output = layers.Softmax(
            name="densenet_probabilities"
        )(logits)

    elif name == "convnext":
        try:
            base = tf.keras.applications.ConvNeXtTiny(
                weights="imagenet",
                include_top=False,
                include_preprocessing=False,
                input_shape=(*IMAGE_SIZE, 3),
            )
        except TypeError:
            base = tf.keras.applications.ConvNeXtTiny(
                weights="imagenet",
                include_top=False,
                input_shape=(*IMAGE_SIZE, 3),
            )

        for layer in base.layers:
            layer.trainable = False

        x = base(input_tensor)
        embedding = layers.GlobalAveragePooling2D(
            name="convnext_embedding"
        )(x)
        x = layers.Dense(
            512,
            kernel_regularizer=keras.regularizers.l2(1e-3),
        )(embedding)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.35)(x)
        x = layers.Dense(
            256,
            kernel_regularizer=keras.regularizers.l2(1e-3),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.25)(x)
        logits = layers.Dense(
            NUM_CLASSES,
            name="convnext_logits",
        )(x)
        output = layers.Softmax(
            name="convnext_probabilities"
        )(logits)

    else:
        raise ValueError(f"Unknown backbone: {name}")

    model = keras.Model(
        input_tensor,
        output,
        name=BACKBONE_DISPLAY[name].replace("-", "_"),
    )
    return model, base


def make_adamw(learning_rate):
    import tensorflow as tf

    try:
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-4,
        )
    except AttributeError:
        return tf.keras.optimizers.experimental.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-4,
        )


def compile_backbone(
    model,
    name,
    learning_rate=None,
):
    import tensorflow as tf
    from tensorflow import keras

    if name == "densenet":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=(
                DENSENET_LR
                if learning_rate is None
                else learning_rate
            )
        )
    elif name == "efficientnet":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=(
                EFFICIENTNET_LR
                if learning_rate is None
                else learning_rate
            )
        )
    elif name == "convnext":
        optimizer = make_adamw(
            CONVNEXT_LR
            if learning_rate is None
            else learning_rate
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
