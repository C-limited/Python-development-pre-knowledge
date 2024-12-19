import threading,time
flag = False  # 是否降落 True表示起飞状态 False表示降落状态
def start():
    print("\n无人机起飞……")
    global flag
    flag = True
    print('嗡嗡嗡',end=' ')

def go():
    n = 10
    print("\n无人机向前飞行……",n*50,"米")
    for i in range(1,n+1):
        if flag:
            print('前进',i*50,'米',end=' ')
            time.sleep(5)
        else:
            break
def stop():
    global flag
    flag = False
    print("\n无人机降落……")
   

if __name__ == '__main__':
    print('-----主线程开始-----')
    tstart = threading.Thread(target=start)
    tstart.start()
    tstart.join()
    while True:
        tstart.join()
        opt = input('\n请选择要执行的动作（1【前进】；其他【降落】）：')
        if opt=='1':
            tgo = threading.Thread(target=go)
            tgo.start()
        else:
            tstop = threading.Thread(target=stop)
            tstop.start()
            break
    
    print('\n-----主线程结束-----')
    
    '''=============不使用线程=======================
    start()
    while True:
        opt = input('\n请选择要执行的动作（1【前进】；其他【降落】）：')
        if opt=='1':
            go()
        else:
            stop()
            break
    '''
