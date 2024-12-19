class Quadrilateral:
    '''四边形类（基类）'''
    def __init__(self):
        print('四条边、四个角')
        print('任意3边和大于第四边')
        print('内角和为360度')
    
        
class Parallelogam(Quadrilateral):           # 定义平行四边形类（派生类)
    def __init__(self):
        print('===== 平行四边形特性 =====')
        super().__init__()  # 调用基类的__init__()方法
        print('对边平行且相等')
        print('对角相等,两邻角互补')

        
class Recangle(Parallelogam):           # 定义矩形类（派生类）
    def __init__(self):
        print('\n========= 矩形特性 ==========')
        print('4个内角都是直角')
        print('对角线相等且互相平分')
        print('既是轴对称图形,也是中心对称图形')
     
parallelogam = Parallelogam()    # 创建类的实例（平行四边形）
recangle = Recangle()    # 创建类的实例（矩形）
