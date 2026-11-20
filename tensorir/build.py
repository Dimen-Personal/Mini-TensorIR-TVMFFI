# 编译IR
import tvm
import tvm.tirx

from tensorir.matmul import MatMulModule     # 从matmul.py文件里导入我们之前定义MatmulMoudule





def build_matmul():
    mod = MatMulModule

    target = "llvm"
    
    executable = tvm.compile(mod, target = target)
    
    return executable





def main():
    mod = MatMulModule
    
    print("====TensorIR Module====")
    print(mod.script())
    
    
    
    
    
    # compile target
    target = "llvm"
    
    print("\n===== Compile Target =====")
    print(target)
    
    
    
    
    
    # compile Tensor IR
    # TensorIR
    #   ↓
    # Lowering
    #   ↓
    # LLVM Code Generation
    #   ↓
    # Executable
    executable = build_matmul()
    print("\n===== Compilation Finished =====")

    print("Executable type:")
    print(type(executable))
    
    
    
    
    
    
    # 取出底层的 runtime module
    runtime_module = executable.mod
    print("\nRuntime module type:")
    print(type(runtime_module))
    
    
    
if __name__ == "__main__":
    main()

    