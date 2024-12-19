# 输入数字退出程序
instr = ''
setlen = 0      # 记录集合的总数
past = set()    # 创建保存取快递人员的集合
# 输入数字退出程序
while instr!= '0':
    instr = input('请输入收到快递人员的名单：')   # 输入数字结束
    setlen = len(past)
    if instr!='0':
        past.add(instr)
        if setlen == len(past):
            print('取快递人员已存在！')
print('需要通知取快递的人员名单：' )
for i in past:
    print(i)

