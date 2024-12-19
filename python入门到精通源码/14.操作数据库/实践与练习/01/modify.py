import pymysql

# 打开数据库连接
db = pymysql.connect(host="localhost", user="root", password="root", database="mrsoft",charset="utf8")
# 使用cursor()方法获取操作游标
cursor = db.cursor()
# 数据列表
data = ("2020-8-20",1)
try:
    # 执行sql语句，修改一条数据
    cursor.execute("UPDATE books SET publish_time = %s WHERE id = %s", data)
    # 提交数据
    db.commit()
except:
    # 发生错误时回滚
    db.rollback()

# 关闭数据库连接
db.close()
