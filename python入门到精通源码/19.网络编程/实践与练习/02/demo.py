# *_* coding： UTF-8 *_*
# 开发团队：明日科技
# 明日学院网站：www.mingrisoft.com
# 开发工具：PyCharm
# 任务：扫描并输出局域网占用的IP地址

import sys, os, socket, string
import threading

list_of_name = []
list_of_ip = []  # 存放结果
thread_pool = []


def showInfo():
    print("""
     命令格式：LAN_ip_hostname -all startip
              LAN_ip_hostname -ip ipaddr
              LAN_ip_hostname -hostname hostname
     说明：-all 扫描局域网中所有IP对应的hostname,需要起始IP，如192.168.0.1
          -ip 获取指定IP的hostname
          -hostname 根据主机名，得到其IP地址
           """)


def lanAll(startip):
    index = startip.rfind('.') # 找最右边的.的索引
    ipfirstpart = startip[0:index + 1] # ip地址中前三位，如192.168.1
    intstart = int(startip[index + 1:])  # ip地址最后一位，转为int型

    f = range(intstart, 255)

    global g_mutex  # 互斥锁。不能定义称全局变量，否则，目标函数不认同
    g_mutex = threading.Lock()  # 初始化互斥锁

    for iplastpart in f:
        targetip = ipfirstpart + str(iplastpart) # 拼接ip
        # 创建线程对象，存为thread。线程要执行的函数由target指定，args指定参数，可以是元组~。线程号从1开始
        thread = threading.Thread(target=lanIp2Name, args=(targetip,))
        thread_pool.append(thread)
        thread.start()

    # 阻塞主线程。collect all threads
    pos = intstart
    for pos in f:
        threading.Thread.join(thread_pool[pos - intstart])

    # 输出结果
    hosts = range(0, len(list_of_name))
    for host in hosts:
        print (list_of_ip[host], '  ====>   ', list_of_name[host])
    print ('Find ', len(list_of_name), ' Hosts.Done!')


def lanIp2Name(ip):
    try:
        (name, aliaslist, addresslist) = socket.gethostbyaddr(ip)
    except:
        return

    global g_mutex  # 再次声明
    g_mutex.acquire()
    ######################受互斥量保护区代码##################################
    list_of_name.append(name)
    list_of_ip.append(ip)
    ########################################################################
    g_mutex.release()


def lanIpToName(ip):
    try:
        (name, aliaslist, addresslist) = socket.gethostbyaddr(ip)
    except:
        return
    print("%s====>%s" % (addresslist,name))

def lanName2Ip(name):
    targetip = socket.gethostbyname(name)
    print (name, "    ====>   ", targetip)


if '__main__' == __name__:
    '''
    sys.argv[]是用来获取命令行参数的，sys.argv[0]表示代码本身文件路径
    array.count(x)  返回出现的x的次数
    '''
    if len(sys.argv) < 3:
        print ("参数错误")
        showInfo()
        exit(1)

    cmds = ['-all', '-ip', '-hostname']

    cmd = sys.argv[1]      # 命令格式
    target = sys.argv[2]   # ip地址

    if 0 == cmds.count(cmd): #判断参数数量
        print ("参数错误啊")
        showInfo()
        exit(1)
    else:
        print ('Start working,Please waiting...')
        if cmd == '-all':   # 输出所有IP和主机名
            lanAll(target)

        elif cmd == '-ip':  # 根据当前IP输出主机名
            lanIpToName(target)

        elif cmd == '-hostname':  # 根据当前主机名输出IP
            lanName2Ip(target)


