price = 6499  # 设定商品价格
n = 0 # 次数
while n<5:  # 5次循环
    n += 1
    value = int(input('请输入这台HUAWEI Mate 40 Pro的价格：')) # 输入数字
    if value > price:
        print('高了')
    elif value <price:
        print('低了！')
    else:
        print('恭喜您，猜对了！')
        break # 退出循环

if n >= 5:
    print('机会已用完！')
