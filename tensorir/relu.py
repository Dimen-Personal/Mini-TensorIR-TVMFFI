from tvm.script import ir as I, tirx as T


@I.ir_module
class ReLUModule:

    @T.prim_func(s_tir = True)   # s_tir 保留各种优化信息，明确 block/axis 等可调度结构
    def relu(
        C: T.Buffer((2, 4), "float32"),
        Y: T.Buffer((2, 4), "float32")
    ):
        for i in range(2):
            for j in range(4):
                
                with T.sblock("relu"):   # 把后面的计算块标记为relu这个计算块
                    vi = T.axis.spatial(2, i)
                    vj = T.axis.spatial(4, j)
                    
                    T.reads(C[vi, vj])
                    T.writes(Y[vi, vj])
                    
                    Y[vi, vj] = T.max(C[vi, vj], T.float32(0))
                    
                    
if __name__ == "__main__":
    print(ReLUModule.script())        