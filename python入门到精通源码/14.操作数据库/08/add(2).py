import pymysql

# 打开数据库连接
db = pymysql.connect(host="localhost", user="root", password="root", database="mrsoft",charset="utf8")
# 使用cursor()方法获取操作游标
cursor = db.cursor()
# 数据列表
data = [("零基础学Python",'Python','79.80','2018-5-20'),
        ("Python从入门到精通",'Python','79.80','2018-10-1'),
        ("Python数据分析从入门到实践",'Python','98.00','2020-6-1'),
        ("Java从入门到精通（第5版）",'Java','69.80','2019-2-1'),
        ("零基础学Java",'Java','69.80','2017-5-18'),
        ]
try:
    # 执行sql语句，插入多条数据
    cursor.executemany("insert into books(name, category, price, publish_time) values (%s,%s,%s,%s)", data)
    # 提交数据
    db.commit()
except:
    # 发生错误时回滚
    db.rollback()

# 关闭数据库连接
db.close()
