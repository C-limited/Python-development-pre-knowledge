while True:
    try:  # 捕获异常
        height = float(input("请输入您的身高（单位为米）："))      # 输入身高，单位：米
        weight = float(input("请输入您的体重（单位为千克)："))     # 输入体重，单位：千克
    except ValueError as e:  # 处理ValueError异常
        print("输入错误：", e)  # 输出错误原因
        continue
    bmi=weight/(height*height)      # 用于计算BMI指数，公式为“体重/身高的平方”
    print("您的BMI指数为："+str(bmi))  # 输出BMI指数
    # 判断身材是否合理
    if bmi<18.5:
        print("您的体重过轻 ~@_@~")
    if bmi>=18.5 and bmi<24.9:
        print("正常范围，注意保持 (-_-)")
    if bmi>=24.9 and bmi<29.9:
        print("您的体重过重 ~@_@~")
    if bmi>=29.9:
        print("肥胖 ^@_@^")
    break
