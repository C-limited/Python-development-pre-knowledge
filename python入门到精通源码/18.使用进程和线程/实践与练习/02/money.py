from threading import Thread,Lock
import time
total=0						   # 卡内初始余额

def task(money):
    global total
    mutex.acquire()				# 上锁
    temp=total					# 赋值给临时变量
    time.sleep(2)				# 休眠2秒
    total=temp+money				# 累加金额 
    print('\n感谢您的爱心奉献，爱心金额%d元'%total)
    mutex.release()				# 释放锁

if __name__ == '__main__':
    mutex=Lock()				# 实例化Lock类
    t_l=[]					# 初始化一个列表
    for i in range(10):
        money = float(input('捐款金额：'))
        t=Thread(target=task,args=(money,))	# 实例化线程类
        t_l.append(t)			# 将线程实例存入列表中
        t.start()				# 创建线程
    for t in t_l:               
        t.join()				# 等待子线程结束    
