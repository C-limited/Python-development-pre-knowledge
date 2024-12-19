usernames = ['qq','wgh','mingri','mr','mrsoft']
username = input('请输入要注册的用户名：')
if username in usernames:
    print('抱歉，该用户名已经被占用！')
else:
    print('恭喜，该用户名可以注册！')
