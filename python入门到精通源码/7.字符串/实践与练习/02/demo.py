# *_* coding： UTF-8 *_*
# 开发团队：明日科技
# 明日学院网站：www.mingrisoft.com
# 格式化输出商品的编号和单价

id = 0
result = ""
while True:
    # 商品信息必须按照商品名称 + 空格 + 品牌 + 空格 + 单价，其中单价只能是数字
    instr = input("请输入商品信息（商品名 单价），按0退出：")
    if instr != '0':
        newlist = instr.split(" ")
        # 去除空数据
        while "" in newlist:
            newlist.remove("")
        prince = float(newlist[1])
        id = id + 1
        result += "%06d\t" % id + "  " + newlist[0] + "  " +  "    " + chr(65509) + " %.2f" % prince + "\n"
    else:
        break
print(result)
