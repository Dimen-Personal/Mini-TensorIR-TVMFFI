# 编译IR
import tvm
import tvm.tirx

from tensorir.matmul import MatMulModule     # 从matmul.py文件里导入我们之前定义MatmulMoudule
from tensorir.relu import ReLUModule





def build_matmul():
    mod = MatMulModule
    target = "llvm"
    executable = tvm.compile(mod, target = target)
    
    return executable



def build_relu():
    executable = tvm.compile(ReLUModule, target = "llvm")
    
    return executable






def main():
    
    print("===== Build MatMul =====")
    matmul_executable = build_matmul()
    print(type(matmul_executable))



    print("\n===== Build ReLU =====")
    relu_executable = build_relu()
    print(type(relu_executable))
    
    
    
if __name__ == "__main__":
    main()

    