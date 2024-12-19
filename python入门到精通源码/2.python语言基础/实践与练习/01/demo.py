print('=========== 程序员计算器 ==========')
number = int(input('请输入一个十进制的整数：'))  # 输入一个十进制的数
b = bin(number)  # 转换为二进制数
o = oct(number)  # 转换为八进制数
h = hex(number)  # 转换为十六进制数
print('十进制数',number,'的二进制数为',b)
print('十进制数',number,'的八进制数为',o)
print('十进制数',number,'的十六进制数为',h)
