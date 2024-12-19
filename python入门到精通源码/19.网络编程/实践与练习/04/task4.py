# *_* coding： UTF-8 *_*
# 开发团队：明日科技
# 明日学院网站：www.mingrisoft.com
# 开发工具：PyCharm
# 任务：获取远程主机的IP地址

import socket
hostname = "www.jd.com"
hostip = socket.gethostbyname(hostname)
print("www.jd.com： "+hostip )
hostname = "www.baidu.com"
hostip = socket.gethostbyname(hostname)
print("www.baidu.com： "+hostip )
hostname = "www.python.com"
hostip = socket.gethostbyname(hostname)
print("www.python.com： " + hostip )
hostname = "www.taobao.com"
hostip = socket.gethostbyname(hostname)
print("www.taobao.com： " + hostip )




