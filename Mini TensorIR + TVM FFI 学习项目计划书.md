# Mini TensorIR + TVM FFI Runtime 项目

## 1. 项目目标

这个项目不追求性能，也不追求完整 AI Framework。

唯一目标是亲手走通：

```text
定义计算
↓
TensorIR 表示计算
↓
TVM 编译
↓
得到可执行函数
↓
通过 TVM FFI 暴露函数
↓
Runtime 调用
↓
得到结果
```

最终希望你能够真正回答：

```text
TensorIR 是什么？
为什么需要 IR？

TVM FFI 是什么？
为什么编译出来以后还需要 FFI？

Compiler 和 Runtime 分别负责什么？

一个 Tensor 到底是怎么从 Runtime
传到编译出来的函数里的？
```

---

# 2. 最终项目效果

最后我们做一个非常小的推理流程：

```text
Input X
[M, K]

   │
   ▼

TensorIR MatMul
X × W

   │
   ▼

TensorIR ReLU

   │
   ▼

TVM Compiler

   │
   ▼

Compiled Function

   │
   ▼

TVM FFI

   │
   ▼

Mini Runtime

   │
   ▼

Output
[M, N]
```

数学表达：

\[
Y = ReLU(XW)
\]

例如：

```text
X:
[2, 3]

W:
[3, 4]

Y:
[2, 4]
```

---

# 3. 为什么选择 MatMul + ReLU

不建议第一版做 Attention。

因为 Attention 会同时引入：

```text
Q/K/V
Softmax
Mask
Reduction
多个中间 Tensor
```

很容易导致你开始研究算法，而不是研究 TensorIR + FFI。

MatMul + ReLU 足够让我们理解：

```text
Tensor
Buffer
Loop
Reduction
Function
Compilation
Runtime
FFI
```

这些才是当前项目真正需要学习的东西。

---

# 4. 项目结构

最终目录可以非常简单：

```text
mini-tvm-runtime/

├── tensorir/
│   ├── matmul.py
│   └── relu.py
│
├── runtime/
│   ├── main.py
│   └── executor.py
│
├── tests/
│   └── test_pipeline.py
│
└── README.md
```

第一版甚至全部写 Python 就可以。

不要一开始引入 C++。

---

# Phase 0 — 建立最基础认知

## 目标

先理解：

```text
Tensor ≠ TensorIR
TensorIR ≠ Runtime
Compiler ≠ Runtime
```

你需要先能够自己解释：

### Tensor

就是：

```text
数据
```

例如：

```text
X =
[
 [1,2,3],
 [4,5,6]
]
```

---

### Operator

对 Tensor 做计算：

```text
MatMul
ReLU
Softmax
```

---

### TensorIR

描述：

```text
这个 Operator 到底怎么算
```

例如 MatMul：

```text
for i
    for j
        C[i,j] = 0

        for k
            C[i,j] += A[i,k] * B[k,j]
```

---

### Compiler

把 TensorIR：

```text
for i
for j
for k
```

变成：

```text
CPU / GPU 可以执行的代码
```

---

### Runtime

运行的时候：

```text
创建 Tensor
↓
找到函数
↓
调用函数
↓
得到结果
```

---

### FFI

解决：

```text
Runtime
如何调用
Compiler 生成的 Function
```

---

# Phase 1 — 用普通 Python 写 MatMul

## 目标

先完全不用 TVM。

自己写：

```python
def matmul(A, B):
    ...
```

例如：

```text
for i
    for j
        for k
```

输入：

```text
A [2,3]

B [3,4]
```

输出：

```text
C [2,4]
```

然后使用 NumPy：

```python
A @ B
```

验证结果。

## 这一阶段要理解

```text
MatMul
本质不是一个神秘 Operator

而是一组：
Loop + Memory Access + Arithmetic
```

---

# Phase 2 — 用 TensorIR 表示 MatMul

这是整个项目第一个核心阶段。

## 目标

把：

```python
C = A @ B
```

变成 TensorIR 中类似：

```text
for i
    for j
        for k
            C[i,j] += A[i,k] * B[k,j]
```

这里重点不要优化。

先只观察：

```text
Buffer
Loop
Block
Reduction axis
```

你需要能够指出：

```text
A Buffer 在哪里

B Buffer 在哪里

C Buffer 在哪里

i/j/k loop 在哪里

k 为什么是 reduction
```

## 阶段成果

你应该能够打印 TensorIR：

```text
print(matmul_ir)
```

然后看见：

```text
for i, j, k
...
```

这时候第一次真正建立：

> TensorIR = 编译器里的程序。

---

# Phase 3 — 编译 TensorIR

## 目标

让 TVM 把：

```text
TensorIR MatMul
```

变成：

```text
CPU 可执行函数
```

第一版只编译 CPU。

不要碰 CUDA。

流程：

```text
TensorIR
↓
TVM build
↓
Module
↓
Function
```

这里重点观察两个东西：

```text
Module
Function
```

你需要开始理解：

> TensorIR 本身不能直接被 Runtime 执行。

它必须先经过：

```text
build
```

变成可以调用的 Function。

---

# Phase 4 — 真正调用编译结果

现在第一次进入 Runtime 世界。

## 目标

创建：

```text
A Tensor
B Tensor
C Tensor
```

然后：

```text
compiled_matmul(A, B, C)
```

得到结果。

与：

```python
numpy_A @ numpy_B
```

比较。

---

## 到这里你应该意识到

前半段：

```text
TensorIR
↓
Build
```

属于：

```text
Compiler
```

后半段：

```text
创建 Tensor
↓
调用 Function
↓
读取结果
```

属于：

```text
Runtime
```

这里就是第一次真正把：

```text
Compiler
```

和：

```text
Runtime
```

区分开。

---

# Phase 5 — 加入 ReLU

现在再增加第二个算子：

\[
Y = ReLU(C)
\]

也就是：

\[
Y_{ij} = max(C_{ij},0)
\]

TensorIR：

```text
for i
    for j
        Y[i,j] = max(C[i,j], 0)
```

于是现在系统变成：

```text
X
↓
MatMul
↓
C
↓
ReLU
↓
Y
```

---

# Phase 6 — 做一个最简单的 Executor

这个阶段和你自己之前的 inference framework 就开始接上了。

写：

```python
class Executor:
    def run(self, op_name, inputs):
        ...
```

例如：

```text
executor.run(
    "matmul",
    [A, B]
)
```

然后：

```text
executor.run(
    "relu",
    [C]
)
```

第一版甚至可以：

```python
if op_name == "matmul":
    ...

elif op_name == "relu":
    ...
```

先不要做复杂 registry。

---

# Phase 7 — 引入 Function Registry

现在开始接近你 NVIDIA 项目里的设计。

把：

```text
if matmul
if relu
```

改成：

```text
Registry

"matmul"
    ↓
MatMul Function

"relu"
    ↓
ReLU Function
```

Runtime：

```text
Executor
↓
Registry.lookup("matmul")
↓
Function
↓
execute
```

这时候你的系统会变成：

```text
             Compiler Side

TensorIR MatMul
      ↓
    build
      ↓
Compiled Function
      │
      │
──────┼────────────────
      │
      ▼
 Function Registry
      │
      ▼

             Runtime Side

Executor
      ↓
lookup("matmul")
      ↓
Function
      ↓
Tensor
```

---

# Phase 8 — 理解 TVM FFI

这里才正式深入 TVM FFI。

不要一开始就研究 FFI internals。

先问一个问题：

> Registry 里面的 Function 为什么可以被 Python / C++ / Runtime 统一调用？

答案开始指向：

```text
TVM FFI
```

你要观察：

```text
Function
Tensor
Module
Object
```

在 Runtime 和 Compiler 之间是怎么传递的。

这时候重点理解：

```text
FFI
不是负责计算。

它只是负责：
“让我可以调用这个 Function”
```

可以把它画成：

```text
Python Runtime

Executor
   │
   │ call
   ▼

TVM FFI Function

   │
   │ ABI
   ▼

Compiled Function

   │
   ▼

CPU
```

---

# Phase 9 — 把 MatMul + ReLU 组成完整 Pipeline

现在最终运行：

```text
Input X
↓
Executor

lookup("matmul")
↓
Compiled TensorIR MatMul
↓
C

lookup("relu")
↓
Compiled TensorIR ReLU
↓
Y
```

验证：

```text
TVM Output
≈
NumPy Output
```

---

# Phase 10 — 最终理解测试

项目完成以后，不看资料回答下面问题：

### Question 1

为什么不能：

```text
TensorIR
直接执行？
```

正确方向：

```text
TensorIR 是程序表示
需要编译成为可执行函数
```

---

### Question 2

TensorIR负责什么？

```text
描述：

Loop
Buffer
Memory Access
Computation
```

---

### Question 3

Runtime负责什么？

```text
Tensor 管理
Function Lookup
Function Invocation
Execution
```

---

### Question 4

TVM FFI负责什么？

```text
让 Runtime
能够以统一方式
调用 Function / Module
和传递 Tensor
```

---

### Question 5

为什么需要 Registry？

因为 Runtime 不应该写：

```text
if matmul:
    调这个地址

if relu:
    调另一个地址
```

可以：

```text
name
↓
lookup
↓
Function
```

---

# 5. 最终完整架构

项目结束以后，你应该能完全理解下面这张图：

```text
                 Compiler
                    │
                    │
         ┌──────────▼──────────┐
         │      TensorIR       │
         │                     │
         │ for i               │
         │   for j             │
         │     for k           │
         │       A*B           │
         └──────────┬──────────┘
                    │
                  build
                    │
                    ▼
         ┌─────────────────────┐
         │ Compiled Function   │
         └──────────┬──────────┘
                    │
                    │
              TVM FFI / ABI
                    │
                    ▼
         ┌─────────────────────┐
         │ Function Registry   │
         └──────────┬──────────┘
                    │
                    │ lookup
                    ▼
         ┌─────────────────────┐
         │      Executor       │
         └──────────┬──────────┘
                    │
                    │ Tensor
                    ▼
         ┌─────────────────────┐
         │        CPU          │
         └─────────────────────┘
```

---

# 6. 暂时不要做的东西

第一版不要加入：

```text
CUDA
Attention
FlashAttention
Multi-head Attention
Autograd
Dynamic Shape
Graph Optimization
Distributed Runtime
```

否则项目会迅速膨胀。

我们现在真正需要学习的只有：

```text
TensorIR
↓
Build
↓
Function
↓
FFI
↓
Runtime
```

---

# 7. 可选 Phase 11 — CUDA

等 CPU 版本完全理解之后，再把同一个：

```text
MatMul
```

从：

```text
LLVM CPU
```

换成：

```text
CUDA GPU
```

然后观察：

```text
Runtime代码
几乎不变

但Compiler生成的底层代码
从CPU变成GPU
```

这一步会让你真正理解：

> 为什么 compiler/runtime abstraction 有价值。

因为 Executor 可以仍然是：

```text
run("matmul")
```

但背后：

```text
今天 → CPU kernel

明天 → CUDA kernel
```

Runtime 不需要完全重写。

---

# 8. 推荐完成顺序

整个项目控制在很小的规模：

```text
Day 1
普通 MatMul
+
TensorIR MatMul

Day 2
Build
+
Runtime 调用

Day 3
ReLU
+
Executor
+
Registry

Day 4
重点研究 TVM FFI
+
整理完整数据流

Optional Day 5
CPU → CUDA
```

实际上代码量可能只有几百行。

---

# 9. 最终项目名称

建议：

```text
Mini TensorIR Runtime
```

或者：

```text
TensorIR-FFI-Learning-Runtime
```

README 第一行就可以写：

> A minimal learning project that demonstrates how TensorIR-compiled operators are exposed through TVM FFI and invoked by a lightweight runtime.

---

# 10. 这个项目真正需要你学会的核心

最后不要以：

```text
“我会写 TensorIR syntax 了”
```

作为完成标准。

真正完成标准应该是你能自己画出来：

```text
Tensor
↓
TensorIR
↓
Compiler
↓
Compiled Function
↓
TVM FFI
↓
Registry
↓
Executor
↓
Runtime
↓
CPU/GPU
```

并且能够解释：

```text
每一层为什么存在。

去掉其中某一层会发生什么。

数据和函数是怎么跨层移动的。
```

如果你达到这个程度，那么你基本就真正理解了 TensorIR + TVM FFI integration。