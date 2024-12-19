# *_* coding： UTF-8 *_*
# 开发团队：明日科技
# 明日学院网站：www.mingrisoft.com
# 输出福彩3D号码

#1.产生一个3D彩票
import random
def ball3d():
    radball = ""
    number = "1234567890"
    for i in range(3):
        radball = radball + str(random.choice(number))
    return str(radball)

print(ball3d())

#2.输出3个手工3D彩票号码和3个随机3D彩票号码
import random

def ball3d():
    radball = ""
    number = "1234567890"
    for i in range(3):
        radball = radball + str(random.choice(number))
    return str(radball)

def inputnew():
    instr = input("请输入一个福彩3D彩票（3位数字）:")  # 请输入一个福彩3D号码(只能为3位数字)
    instr = instr.strip()
    return instr

def inputbox():
    cxstr = inputnew()
    if len(cxstr) == 3 and cxstr.isdigit():  # 输入数据的长度不为零
        return cxstr
    else:
        print("输入错误，请重新输入3位数字的3D彩票:")  # 请输入一个福彩3D号码(只能为3位数字)

inball = []
while len(inball) < 3:
    inball.append(inputbox())
    if None in inball:
        inball.remove(None)
inball.append(ball3d())
inball.append(ball3d())
inball.append(ball3d())
print("您需要的3D彩票已生成:")
print(("\n").join(inball))



