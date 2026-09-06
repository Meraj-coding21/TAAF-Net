import time
import numpy as np


def force_value(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            force_value(item)
    elif hasattr(value, "numpy"):
        value.numpy()


def benchmark_function(
    function,
    warmup=50,
    iterations=500,
):
    for _ in range(warmup):
        force_value(function())

    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        result = function()
        force_value(result)
        times.append(
            (time.perf_counter() - start)
            * 1000.0
        )

    times = np.asarray(
        times,
        dtype=np.float64,
    )

    return {
        "Latency_ms": float(times.mean()),
        "Latency_std_ms": float(
            times.std(ddof=1)
        ),
        "Throughput_FPS": float(
            1000.0 / times.mean()
        ),
    }


def gpu_peak_mib(function):
    import tensorflow as tf

    if not tf.config.list_physical_devices(
        "GPU"
    ):
        return np.nan

    try:
        tf.config.experimental.reset_memory_stats(
            "GPU:0"
        )
        force_value(function())
        info = tf.config.experimental.get_memory_info(
            "GPU:0"
        )
        return float(
            info["peak"] / (1024 ** 2)
        )
    except Exception:
        return np.nan


def estimate_gflops(
    model,
    input_shapes,
):
    try:
        import tensorflow as tf
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph,
        )

        specs = [
            tf.TensorSpec(
                [1, *shape],
                tf.float32,
            )
            for shape in input_shapes
        ]

        @tf.function
        def forward(*args):
            inputs = (
                list(args)
                if len(args) > 1
                else args[0]
            )
            return model(
                inputs,
                training=False,
            )

        concrete = (
            forward.get_concrete_function(
                *specs
            )
        )
        _, graph_def = (
            convert_variables_to_constants_v2_as_graph(
                concrete
            )
        )

        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(
                graph_def,
                name="",
            )
            options = (
                tf.compat.v1.profiler
                .ProfileOptionBuilder
                .float_operation()
            )
            profile = (
                tf.compat.v1.profiler.profile(
                    graph,
                    options=options,
                )
            )

            if profile is None:
                return np.nan

            return float(
                profile.total_float_ops
                / 2
                / 1e9
            )

    except Exception:
        return np.nan
