print('='*10,'加法器','='*10)
one = '' # 第一个加数
two = '' # 第二个加数
while True:
    if type(one) == str:
        one = input('请输入第一个加数：')
        try:
            one = int(one)
        except:
            try:
                one = float(one)
            except:
                print('输入的不是有效的数值！')
                continue  # 重新执行
    if type(two) == str:
        two = input('请输入第二个加数：')
        try:
            two = int(two)
            break # 跳出循环
        except:
            try:
                two = float(two)
                break # 跳出循环
            except:
                print('输入的不是有效的数值！')
                continue  # 重新执行
print(one + two)  # 输出相加的结果
