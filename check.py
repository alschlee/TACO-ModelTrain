import onnx

model = onnx.load("/Volumes/My Passport/TACO/yolov5/runs/train/taco_yolov5s_model2/weights/best.onnx")

for node in model.graph.node:
    if node.op_type == "Split":
        print(f"Node Name: {node.name}")
        print(f"Inputs: {node.input}")
        print(f"Outputs: {node.output}")
        for attr in node.attribute:
            print(f"Attribute - Name: {attr.name}, Value: {attr.i if hasattr(attr, 'i') else attr.f}")
        print("-" * 50)