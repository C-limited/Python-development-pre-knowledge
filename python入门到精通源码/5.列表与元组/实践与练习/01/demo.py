months = ['January','February','March','April','May','June','July','Aygust', 'September','October','November','December']
month = int(input('请输入要查询的月份（1~12）：'))
if month in range(1,13):
    print(month,'月的英文是',months[month-1])
else:
    print('您输入的月份不合法！')
