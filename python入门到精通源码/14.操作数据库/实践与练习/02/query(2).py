import pymysql

# 打开数据库连接
db = pymysql.connect(host="localhost", user="root", password="root", database="mrsoft",charset="utf8")
# 使用cursor()方法获取操作游标
cursor = db.cursor()
# 数据列表
data = ("Python")
# 执行sql语句，修改一条数据
cursor.execute("SELECT * FROM books WHERE category = %s", data)
myresult = cursor.fetchall()

for x in myresult:
  print(x)

# 关闭数据库连接
db.close()
