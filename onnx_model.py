import onnx
from onnx import helper

model = onnx.load("/Volumes/My Passport/TACO/yolov5/runs/train/taco_yolov5s_model2/weights/best.onnx")

for node in model.graph.node:
    if node.op_type == "Split":
        axis_found = False
        for attr in node.attribute:
            if attr.name == "axis":
                attr.i = 2
                axis_found = True
        if not axis_found:
            node.attribute.extend([helper.make_attribute("axis", 2)])
            
        split_found = False
        for attr in node.attribute:
            if attr.name == "split":
                split_found = True
                attr.ints[:] = [1, 1, 1]
        if not split_found:
            node.attribute.extend([helper.make_attribute("split", [1, 1, 1])])

reshape_node = helper.make_node(
    'Reshape',
    inputs=["output0"],
    outputs=["output0_reshaped"],
    shape=[1, 25200, 65, 1]
)
model.graph.node.append(reshape_node)

# 그래프의 출력 노드를 업데이트
for output in model.graph.output:
    if output.name == "output0":
        # 출력 텐서 이름 업데이트
        output.name = "output0_reshaped"
        del output.type.tensor_type.shape.dim[:]
        for d in [1, 25200, 65, 1]:
            new_dim = output.type.tensor_type.shape.dim.add()
            new_dim.dim_value = d

# 수정된 모델 저장
onnx.save(model, "/Volumes/My Passport/TACO/yolov5/runs/train/taco_yolov5s_model2/weights/best_fixed_with_split_and_reshape_v3.onnx")
