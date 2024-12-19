# 加法
def add(opt1,opt2):
    if (type(eval(opt1)) == float or type(eval(opt1)) == int) and (type(eval(opt2)) == float or type(eval(opt2)) == int):
        return eval(opt1) + eval(opt2)  # 进行加法运算
    else:
        return '您的输入有误！'
# 减法
def sub(opt1,opt2):
    if (type(eval(opt1)) == float or type(eval(opt1)) == int) and (type(eval(opt2)) == float or type(eval(opt2)) == int):
        return eval(opt1) - eval(opt2)  # 进行减法运算
    else:
        return '您的输入有误！'

def mul(opt1,opt2):
    if (type(eval(opt1)) == float or type(eval(opt1)) == int) and (type(eval(opt2)) == float or type(eval(opt2)) == int):
        return eval(opt1) * eval(opt2)  # 进行乘法运算
    else:
        return '您的输入有误！'
    
def div(opt1,opt2):
    if (type(eval(opt1)) == float or type(eval(opt1)) == int) and (type(eval(opt2)) == float or type(eval(opt2)) == int):
        if eval(opt2)== 0 :
            return '除数不能为零。'
        else:
            return eval(opt1) / eval(opt2)  # 进行加法运算
    else:
        return '您的输入有误！'
