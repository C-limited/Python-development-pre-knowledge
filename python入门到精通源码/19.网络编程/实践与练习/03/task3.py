# *_* coding： UTF-8 *_*
# 开发团队：明日科技
# 明日学院网站：www.mingrisoft.com
# 开发工具：PyCharm
# 任务：输出本地计算机名称与本地计算机的IP地址

import socket
hostname = socket.gethostname()
print("你的计算机名称为： "+hostname )
hostip = socket.gethostbyname(hostname)
print("你的计算机IP地址为： " + hostip )





