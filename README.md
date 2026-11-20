# Mini TensorIR + TVM FFI

## 运行 TensorIR 示例

已在 Apple Silicon macOS 和 Python 3.11 上验证。在仓库根目录运行：

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tensorir/matmul.py
```

脚本打印一个 `2×3 @ 3×4` 矩阵乘法的 TensorIR，包含两个空间轴和一个归约轴。
