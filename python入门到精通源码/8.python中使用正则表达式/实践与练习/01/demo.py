import re                                             # 导入Python的re模块
pattern = r'天猫|当当|唯一|神效'            # 模式字符串
about = '质量上乘，价格实惠，比天猫、当当还便宜。'
print('\n原内容：',about)
sub = re.sub(pattern, '@_@', about)         # 进行模式替换
print('替换后：',sub)
