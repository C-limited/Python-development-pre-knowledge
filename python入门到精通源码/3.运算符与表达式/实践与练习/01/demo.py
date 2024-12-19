# *_* coding： UTF-8 *_*
# 开发团队：明日科技
# 明日学院网站：www.mingrisoft.com
# 计算汽车平均油耗

money = int(input("请输入加油的钱数：\n"))
s = int(input("请输入行驶的公里数：\n"))
p = money/s
print("您车辆的每公里花费为：",p,"元")
# 假设汽油的价格为6.27元
x=money/6.27  # 计算所用汽油的升数
print('您汽车的百公里油耗为：',x/s*100)# 油耗



