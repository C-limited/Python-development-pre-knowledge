import os

def formatTime(longtime):
    '''格式化日期时间的函数
       longtime：要格式化的时间
    '''
    import time                                 # 导入时间模块
    return time.strftime('%Y-%m-%d %H：%M：%S',time.localtime(longtime))

    '''如果在PyCharm中运行出现“”错误，那么将上面的代码替换为以下代码：
    return time.strftime('%Y-%m-%d %H{h}%M{m}%S',time.localtime(longtime)).format(h='：',m='：')
    '''


path = r'pic'
#获取该文件夹下所有文件，并存入列表
fileList=os.listdir(path)
for i in fileList:
    newpath = os.path.join(path,i)
    fileinfo = os.stat(newpath) # 获取文件信息
    createtime = formatTime(fileinfo.st_mtime)  # 获取拍摄时间
    filename = os.path.basename(newpath)
    dst = os.path.join(path,createtime+'_'+filename)
    if os.path.exists(newpath):    # 判断文件是否存在
        os.rename(newpath,dst)
print('重命名完成。')


