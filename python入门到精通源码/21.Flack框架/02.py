from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/user/<username>')
def show_user_profile(username):
    # 显示该用户名的用户信息
    return 'User %s' % username

@app.route('/post/<int:post_id>')
def show_post(post_id):
    # 根据ID显示文章，ID是整型数据
    if post_id == 10:
        content = '腹有读书气自华'
    else:
        content = '文章不存在！'
    return 'ID【 %d 】内容为：%s' % (post_id,content)

if __name__ == '__main__':
    app.run()
