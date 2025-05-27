import os

import torch
import torch_tensorrt


def export_torch(model: torch.nn.Module, path: str, input_: torch.Tensor) -> None:
    exported_program = torch.export.export(
        model.eval().cuda(), (input_.to(torch.float).cuda(),), strict=True
    )
    exported_training = torch.export.export_for_training(
        model.eval().cuda(), (input_.to(torch.float).cuda(),), strict=True
    )

    torch.export.save(exported_program, os.path.join(path, "model.pt2"))
    torch.export.save(exported_training, os.path.join(path, "model_training.pt2"))
    # saved_exported_program = torch.export.load("exported_program.pt2")


def export_tensorrt(model: torch.nn.Module, path: str, input_: torch.Tensor) -> None:
    trt_gm = torch_tensorrt.compile(
        model.eval().cuda(), ir="dynamo", inputs=[input_.to(torch.float).cuda()]
    )

    torch_tensorrt.save(trt_gm, os.path.join(path, "model.ep"), inputs=[input_])

    # Later, you can load it and run inference
    # model = torch.export.load("trt.ep").module()


def export_onnx(model: torch.nn.Module, path: str, input_: torch.Tensor) -> None:
    onnx_program = torch.onnx.export(
        model.eval().cuda(),
        (input_.to(torch.float).cuda(),),
        dynamo=True,
        optimize=True,
        verify=True,
    )

    assert onnx_program is not None

    onnx_program.save(os.path.join(path, "model.onnx"))
