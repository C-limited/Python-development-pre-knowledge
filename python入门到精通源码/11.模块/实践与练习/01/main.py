from calculator import *  # 导入calculator模块下的全部定义 
opt1 = input('输入第一个数:')
opt2 = input('输入第二个数:')
print('='*8,'开始计算','='*8)
print('加法的结果：',add(opt1,opt2))   # 进行加法运算
print('减法的结果：',sub(opt1,opt2))   # 进行减法运算
print('乘法的结果：',mul(opt1,opt2))   # 进行乘法运算
print('除法的结果：',div(opt1,opt2))   # 进行除法运算

