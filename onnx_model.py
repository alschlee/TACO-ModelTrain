import onnx

model = onnx.load("/Volumes/My Passport/TACO/yolov5/runs/train/taco_yolov5s_model2/weights/best.onnx")

for node in model.graph.node:
    if node.op_type == "Split":
        for attr in node.attribute:
            if attr.name == "axis":
                attr.i = 4
        if not any(attr.name == "split" for attr in node.attribute):
            new_attr = onnx.helper.make_attribute("split", [1, 1, 1])
            node.attribute.extend([new_attr])

for output in model.graph.output:
    if output.name == "output0":
        output.name = "output0_fixed"
        del output.type.tensor_type.shape.dim[:]
        for d in [1, 96000, 1, 1]:
            new_dim = output.type.tensor_type.shape.dim.add()
            new_dim.dim_value = d

onnx.save(model, "/Volumes/My Passport/TACO/yolov5/runs/train/taco_yolov5s_model2/weights/final_modified_output.onnx")
