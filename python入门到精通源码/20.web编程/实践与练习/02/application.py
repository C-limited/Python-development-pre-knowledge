#coding=utf-8
def app(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])     # 响应信息
    file_name = environ['PATH_INFO'][1:] or 'index.html'        # 获取url参数
    HTML_ROOT_DIR = './Views/'  # 设置HTML文件目录
    try:
        file = open(HTML_ROOT_DIR + file_name, 'rb')  # 打开文件
    except IOError:
        response_body = '404</br>要访问的网页不存在!'     # 如果异常，返回404
    else:
        file_data = file.read() # 读取文件内容
        file.close()            # 关闭文件
        response_body = file_data.decode('utf-8') # 构造响应数据

    return [response_body.encode('gbk')] # 返回数据
