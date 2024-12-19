import requests     # 导入请求模块
from bs4 import BeautifulSoup  # 导入BeautifulSoup库

url = 'https://news.sina.com.cn/'     # 创建请求地址
# 创建模拟浏览器的请求头信息
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36'}
response = requests.get(url=url,headers=headers)   # 发送网络请求
response.encoding='utf-8'                          # 设置编码方式
# 创建一个BeautifulSoup对象，获取页面正文
soup = BeautifulSoup(response.text, features="lxml")
print(soup.prettify())     # 打印格式化后的代码

