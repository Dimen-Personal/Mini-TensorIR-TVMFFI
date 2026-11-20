from tvm.script import ir as I, tirx as T


@I.ir_module
class MatMulModule:
    @T.prim_func(s_tir=True)
    def matmul(
        A: T.Buffer((2, 3), "float32"),
        B: T.Buffer((3, 4), "float32"),
        C: T.Buffer((2, 4), "float32"),
    ):
        for i in range(2):
            for j in range(4):
                for k in range(3):
                    with T.sblock("matmul"):
                        # 块轴描述逻辑维度，循环变量提供实际坐标。
                        vi = T.axis.spatial(2, i)
                        vj = T.axis.spatial(4, j)
                        vk = T.axis.reduce(3, k)
                        T.reads(A[vi, vk], B[vk, vj])
                        T.writes(C[vi, vj])
                        with T.init():
                            C[vi, vj] = T.float32(0)
                        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


if __name__ == "__main__":
    print(MatMulModule.script())
