import re                                     # 导入Python的re模块
pattern = re.compile(r'([A-Za-z0-9_\-\.])+@([A-Za-z0-9_\-\.])+.([A-Za-z]{2,4})')
search = pattern.finditer('明日科技：mingrisoft@mingrisoft.com 琦琦：84978981@qq.com '
                          '无语：123456789@163.com 可可：987654321@192.168.1.66.com')
for item in search:
    print(item.group())

