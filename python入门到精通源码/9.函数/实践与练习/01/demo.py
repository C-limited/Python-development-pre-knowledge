def change(money):  # 编写转换函数
    GBP=str(money)+ '人民币 = '+str(format(money *0.1126,'.2f')) +' 英镑'
    USD=str(money)+ '人民币 = '+str(format(money *0.1546,'.2f'))+' 美元'
    EUR=str(money)+ '人民币 = '+str(format(money *0.1284,'.2f')) +' 欧元'
    JPY=str(money)+ '人民币 = '+str(format(money *16.2935,'.2f')) +' 日元'
    s=str(GBP)+'\n'+str(USD)+'\n'+str(EUR)+'\n'+str(JPY)+'\n'
    print(s)  # 输出转换结果

instr=input('请输入要兑换的人民币金额：')  # 输入人民币金额
change(int(instr))  # 调用函数转换
