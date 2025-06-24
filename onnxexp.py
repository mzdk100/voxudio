from torch import onnx

OPSET_VERSION = 20


def export_tcc(model, output_path, src_audio, src_se, tgt_se):
    onnx.export(
        model,
        f=output_path,
        args=(src_audio, src_se, tgt_se),
        input_names=(
            "src_audio",
            "src_se",
            "tgt_se",
        ),
        output_names=("audio",),
        dynamic_axes={
            "src_audio": {0: "src_samples", 1: "src_channels"},
            "src_se": {0: "src_channels"},
            "tgt_se": {0: "tgt_channels"},
        },
        do_constant_folding=True,  # whether to execute constant folding for optimization
        opset_version=OPSET_VERSION,
    )


def export_see(model, output_path, audio):
    onnx.export(
        model,
        f=output_path,
        args=(audio,),
        input_names=("audio",),
        output_names=("se",),
        dynamic_axes={
            "audio": {0: "samples", 1: "channels"},
        },
        do_constant_folding=True,  # whether to execute constant folding for optimization
        opset_version=OPSET_VERSION,
    )
