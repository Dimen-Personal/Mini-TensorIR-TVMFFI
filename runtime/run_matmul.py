import numpy as np
import tvm

from tensorir.build import build_matmul



def main():
    
    # Compile Tendor IR
    executable = build_matmul()
    runtime_module = executable.mod

    
    
    # 从runtime module中找到变异好的 matmul function
    matmul_func = runtime_module["matmul"]
    print("===== Runtime Function =====")
    print(type(matmul_func))
    
    
    
    
    # 真实输入
    np_A = np.array([
        [1,2,3],
        [4,5,6]
        ],
        
        dtype = "float32"
    )
    
    np_B = np.array([
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12]
        ],
        
        dtype = "float32"
    )
    
    
    
    
    
    # Step 4
    # NumPy Array
    #      ↓
    # TVM Runtime Tensor
    A = tvm.runtime.tensor(np_A)
    B = tvm.runtime.tensor(np_B)
    
    
    
    
    
    
    # Step 5
    # 创建输出 Tensor C
    #
    # Shape:
    # [2,3] x [3,4]
    #      ↓
    # [2,4]
    np_C = np.zeros(
        (2, 4),
        dtype = "float32"
    )
    C = tvm.runtime.tensor(np_C)
    
    print("\n===== Input A =====")
    print(A.numpy())

    print("\n===== Input B =====")
    print(B.numpy())
    
    
    
    
    
    
    # Step 6
    # 调用编译出来的函数
    #
    # Runtime Tensor
    #      ↓
    # Compiled Function
    #      ↓
    # CPU
    matmul_func(A,B,C)
    
    
    
    
    
    
    
    # Step 7
    tvm_result = C.numpy()     # 转回NP数组
    
    print("\n===== TVM Result =====")
    print(tvm_result)
    
    
    
    
    
    # NUMPY 验证
    expected = np_A @ np_B
    print("\n===== NumPy Reference =====")
    print(expected)
    
    
    
    
    
    # Correctness Check
    correct = np.allclose(tvm_result, expected)
    
    print("\n===== Correct =====")
    print(correct)
    
    
if __name__ == "__main__":
    main()
    