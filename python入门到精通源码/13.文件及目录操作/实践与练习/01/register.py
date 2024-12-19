while True:
    print('*'*35)
    print(' '*10,'用户注册',' '*20)
    print('*'*35)
    name = input('用户名称：')
    pwd = input('密码：')
    repwd = input('确认密码：')
    realname = input('真实姓名：')
    email = input('E-mail地址：')
    question = input('找回密码的问题：')
    answer = input('答案：')
    if name != '' and pwd !='' and realname != '':
        if pwd !=repwd: # 判断两次输入的密码是否一致
            print('两次输入的密码不一致！请重新输入！')
            continue
        else:  # 保存用户注册信息
            with open('user.txt','a')as file:
                file.write('用户名称：'+name+'\n')
                file.write('用户密码：'+pwd+'\n')
                file.write('真实姓名：'+realname+'\n')
                file.write('E-mail地址：'+email+'\n')
                file.write('找回密码的问题：'+question+'\n')
                file.write('答案：'+answer+'\n')
                print('注册成功！')
                break   # 退出循环

