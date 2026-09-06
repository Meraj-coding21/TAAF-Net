from pathlib import Path
import numpy as np

from .backbones import BACKBONE_DIMS, make_adamw
from .config import (
    FUSION_BATCH_SIZE,
    FUSION_EPOCHS,
    FUSION_LR,
    LABEL_SMOOTHING,
    NUM_CLASSES,
    SEED,
)
from .evaluation import sample_weights


def get_static_channel_gate_class():
    import tensorflow as tf
    from tensorflow.keras import layers

    @tf.keras.utils.register_keras_serializable(
        package="TAAF"
    )
    class StaticChannelGate(layers.Layer):
        def build(self, input_shape):
            self.gate_logits = self.add_weight(
                name="gate_logits",
                shape=(int(input_shape[-1]),),
                initializer="zeros",
                trainable=True,
            )

        def call(self, inputs):
            return inputs * tf.sigmoid(self.gate_logits)

    return StaticChannelGate


StaticChannelGate = None


def _static_gate():
    global StaticChannelGate
    if StaticChannelGate is None:
        StaticChannelGate = get_static_channel_gate_class()
    return StaticChannelGate


def build_fusion_model(
    branches=("efficientnet", "densenet", "convnext"),
    texture_mode="full",
    fusion_type="attention",
):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = []
    projected = []

    for name in branches:
        inp = keras.Input(
            shape=(BACKBONE_DIMS[name],),
            name=f"{name}_embedding_input",
        )
        inputs.append(inp)

        x = layers.Dense(
            256,
            use_bias=False,
            name=f"{name}_projection",
        )(inp)
        x = layers.BatchNormalization(
            name=f"{name}_projection_bn"
        )(x)
        x = layers.ReLU(
            name=f"{name}_projection_relu"
        )(x)
        x = layers.Dropout(
            0.30,
            name=f"{name}_projection_dropout",
        )(x)
        projected.append(x)

    texture_dim = {
        "full": 26,
        "lbp": 10,
        "glcm": 16,
        "none": None,
    }[texture_mode]

    if texture_dim is not None:
        texture_input = keras.Input(
            shape=(texture_dim,),
            name="texture_input",
        )
        inputs.append(texture_input)
        texture_projection = layers.Dense(
            64,
            name="texture_projection",
        )(texture_input)
        texture_projection = layers.ReLU(
            name="texture_projection_relu"
        )(texture_projection)
        texture_projection = layers.Dropout(
            0.30,
            name="texture_projection_dropout",
        )(texture_projection)
        projected.append(texture_projection)

    v_fused = layers.Concatenate(
        name="v_fused"
    )(projected)

    fused_dim = (
        256 * len(branches)
        + (64 if texture_dim is not None else 0)
    )

    if fusion_type == "attention":
        hidden_dim = fused_dim // 2
        hidden = layers.Dense(
            hidden_dim,
            activation="relu",
            name="attention_hidden",
        )(v_fused)
        alpha = layers.Dense(
            fused_dim,
            activation="sigmoid",
            name="alpha",
        )(hidden)
        v_final = layers.Multiply(
            name="v_final"
        )([v_fused, alpha])

    elif fusion_type == "uniform":
        gate_class = _static_gate()
        gated = gate_class(
            name="static_channel_gate"
        )(v_fused)
        v_final = layers.Activation(
            "linear",
            name="v_final",
        )(gated)

    elif fusion_type == "concat":
        v_final = layers.Activation(
            "linear",
            name="v_final",
        )(v_fused)

    else:
        raise ValueError(
            f"Unknown fusion type: {fusion_type}"
        )

    x = layers.Dense(
        512,
        use_bias=False,
        name="meta_dense_512",
    )(v_final)
    x = layers.BatchNormalization(
        name="meta_bn_512"
    )(x)
    x = layers.ReLU(
        name="meta_relu_512"
    )(x)
    x = layers.Dropout(
        0.40,
        name="meta_dropout_512",
    )(x)

    x = layers.Dense(
        256,
        use_bias=False,
        name="meta_dense_256",
    )(x)
    x = layers.BatchNormalization(
        name="meta_bn_256"
    )(x)
    x = layers.ReLU(
        name="meta_relu_256"
    )(x)
    x = layers.Dropout(
        0.30,
        name="meta_dropout_256",
    )(x)

    logits = layers.Dense(
        NUM_CLASSES,
        name="logits",
    )(x)
    probabilities = layers.Softmax(
        name="probabilities"
    )(logits)

    return keras.Model(
        inputs,
        probabilities,
        name="TAAF_Net",
    )


def compile_fusion(model):
    from tensorflow import keras

    model.compile(
        optimizer=make_adamw(FUSION_LR),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING,
        ),
        metrics=["accuracy"],
    )


def fusion_input_list(
    embeddings,
    texture,
    branches,
):
    values = [
        embeddings[name]
        for name in branches
    ]

    if texture is not None:
        values.append(texture)

    return values


def train_fusion_model(
    train_embeddings,
    val_embeddings,
    y_train,
    y_val,
    train_texture,
    val_texture,
    branches=("efficientnet", "densenet", "convnext"),
    texture_mode="full",
    fusion_type="attention",
    save_path=None,
    verbose=1,
):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)

    model = build_fusion_model(
        branches=branches,
        texture_mode=texture_mode,
        fusion_type=fusion_type,
    )
    compile_fusion(model)

    train_inputs = fusion_input_list(
        train_embeddings,
        train_texture,
        branches,
    )
    val_inputs = fusion_input_list(
        val_embeddings,
        val_texture,
        branches,
    )

    y_train_onehot = np.eye(
        NUM_CLASSES,
        dtype=np.float32,
    )[np.asarray(y_train, dtype=int)]
    y_val_onehot = np.eye(
        NUM_CLASSES,
        dtype=np.float32,
    )[np.asarray(y_val, dtype=int)]

    history = model.fit(
        train_inputs,
        y_train_onehot,
        validation_data=(
            val_inputs,
            y_val_onehot,
        ),
        sample_weight=sample_weights(y_train),
        epochs=FUSION_EPOCHS,
        batch_size=FUSION_BATCH_SIZE,
        shuffle=True,
        verbose=verbose,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        model.save(save_path)

    return model, history.history
