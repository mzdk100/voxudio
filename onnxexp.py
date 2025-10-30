from onnxoptimizer import optimize, get_fuse_and_elimination_passes
from onnx import load, save
from onnxsim.onnx_simplifier import simplify
from onnxslim import slim
from torch import onnx

OPSET_VERSION = 20
passes = get_fuse_and_elimination_passes()

def optim(model_path, use_slim=True):
    model = load(model_path)
    model = optimize(model, passes=passes)
    model, _ = simplify(model)
    if use_slim:
        model = slim(model)
    save(model, model_path)



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
        dynamo=False,
        do_constant_folding=True,  # whether to execute constant folding for optimization
        opset_version=OPSET_VERSION,
    )
    optim(output_path, use_slim=False)


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
        dynamo=False,
        do_constant_folding=True,  # whether to execute constant folding for optimization
        opset_version=OPSET_VERSION,
    )
    optim(output_path)

