
import random,threading,time
flag = False  # 是否降落 True表示降落状态 False表示起飞状态
def start():
    print("\n无人机起飞……")
    global flag
    flag = True
    print('嗡嗡嗡',end=' ')

def go():
    n = random.randint(1,10) 
    print("\n无人机向前飞行……",n*50,"米")
    for i in range(1,n+1):
        if flag:
            print('前进',i*50,'米',end=' ')
            time.sleep(5)
        else:
            break
def rotate():
    n = random.randint(1,4)    
    print("\n无人机旋转……",n,"次")
    for i in range(1,n+1):
        if flag:
            print('旋转',i*90,'度',end=' ')
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
##    start()
    while True:
        tstart.join()
        opt = input('\n请选择要执行的动作（1【前进】；2【旋转】；其他【降落】）：')
        if opt=='1':
            
            tgo = threading.Thread(target=go)
            tgo.start()
        elif opt == '2':

            trotate = threading.Thread(target=rotate)
            trotate.start()
        else:
            tstop = threading.Thread(target=stop)
            tstop.start()
            break
    
    print('\n-----主线程结束-----')
