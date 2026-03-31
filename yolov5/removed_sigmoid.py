import onnx
from onnx import shape_inference

IN_PATH  = "best_640.onnx"              # 換成你的檔名
OUT_PATH = "model_no_sigmoid.onnx"
MAX_HOPS = 6
TAIL_CONSUMERS = {"Concat","Reshape","Transpose","Slice","Add","Sub","Mul","Div","Softmax"}

m = onnx.load(IN_PATH)
m = shape_inference.infer_shapes(m)

# 建 producer / consumers 索引
producer  = {}  # tensor_name -> node
consumers = {}  # tensor_name -> [nodes]
for n in m.graph.node:
    for y in n.output:
        producer[y] = n
    for x in n.input:
        consumers.setdefault(x, []).append(n)

graph_outputs = [o.name for o in m.graph.output]

# 從輸出往回 BFS，收集「尾端路徑上的節點」（用 id 去重）
def back_reachable_nodes(max_hops=MAX_HOPS):
    q = [(t, 0) for t in graph_outputs]
    seen_tensors = set()
    node_ids = set()
    nodes = []
    while q:
        t, d = q.pop(0)
        if t in seen_tensors or d > max_hops:
            continue
        seen_tensors.add(t)
        p = producer.get(t)
        if not p:
            continue
        pid = id(p)
        if pid not in node_ids:
            node_ids.add(pid)
            nodes.append(p)
            for x in p.input:
                q.append((x, d+1))
    return nodes

tail_nodes = back_reachable_nodes()

def is_tail_sigmoid(n):
    if n.op_type != "Sigmoid":
        return False
    # 其輸出必須往下游接到「尾端常見算子」（避免改到 SiLU 內部）
    for y in n.output:
        for c in consumers.get(y, []):
            if c.op_type in TAIL_CONSUMERS:
                return True
    return False

targets = [n for n in tail_nodes if is_tail_sigmoid(n)]
print(f"[Info] found {len(targets)} tail Sigmoid(s):", [t.name or "(no-name)" for t in targets])

patched = 0
for n in targets:
    n.op_type = "Identity"
    # 正確清空 attribute（protobuf repeated field 不能切片賦值）
    while n.attribute:
        n.attribute.pop()
    patched += 1

onnx.checker.check_model(m)
onnx.save(m, OUT_PATH)
print(f"[Done] patched {patched} Sigmoid(s) -> Identity, saved to {OUT_PATH}")
