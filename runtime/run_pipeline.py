import numpy as np
import tvm

from tensorir.build import (
    build_matmul,
    build_relu
)



def main():
    
    matmul_executable = build_matmul()
    relu_executable = build_relu()
    
    
    matmul_module = (matmul_executable.mod)
    relu_module = (relu_executable.mod)
    
    
    
    matmul_func = (matmul_module["matmul"])
    relu_func = (relu_module["relu"])
    
    
    
    
    np_A = np.array(
        [[1, -2, 3],
         [-4, 5, -6]],
        
        dtype = "float32" 
    )
    
    np_B = np.array(
        [[1, 2, -3, 4],
         [5, -6, 7, -8],
         [-9, 10, 11, -12]],
        
        dtype = "float32"
    )
    
    
    
    
    A = tvm.runtime.tensor(np_A)
    B = tvm.runtime.tensor(np_B)
    
    
    
    
    
    C = tvm.runtime.tensor(
        np.zeros((2, 4), dtype = "float32")
    )
    
    
    
    
    Y = tvm.runtime.tensor(
        np.zeros((2, 4), dtype = "float32")
    )
    
    
    
    
    matmul_func(A, B, C)
    
    
    print("===== MatMul Result C =====")
    print(C.numpy())
    
    
    
    
    
    relu_func(C, Y)
    
    
    
    
    print("\n ===== ReLU Result Y =====")
    tvm.result = Y.numpy()
    
    print(tvm.result)
    
    
if __name__ == "__main__":
    main()
    
    