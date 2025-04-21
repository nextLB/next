
"""
https://www.cnblogs.com/TS86/p/18540326#:~:text=%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8Flask%E7%BC%96%E5%86%99%E4%B8%80%E4%B8%AA%E7%BD%91%E7%AB%99%201%20%E4%B8%80%E3%80%81%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8Flask%E7%BC%96%E5%86%99%E4%B8%80%E4%B8%AA%E7%BD%91%E7%AB%99%20%EF%BC%88%E4%B8%80%EF%BC%89%E5%AE%89%E8%A3%85Flask%20%E9%A6%96%E5%85%88%EF%BC%8C%E6%88%91%E4%BB%AC%E9%9C%80%E8%A6%81%E7%A1%AE%E4%BF%9D%E6%88%91%E4%BB%AC%E7%9A%84Python%E7%8E%AF%E5%A2%83%E4%B8%AD%E5%AE%89%E8%A3%85%E4%BA%86Flask%E3%80%82%20%E6%88%91%E4%BB%AC%E5%8F%AF%E4%BB%A5%E4%BD%BF%E7%94%A8pip%EF%BC%88Python%E7%9A%84%E5%8C%85%E7%AE%A1%E7%90%86%E5%99%A8%EF%BC%89%E6%9D%A5%E5%AE%89%E8%A3%85%E5%AE%83%E3%80%82%20bash%E5%A4%8D%E5%88%B6%E4%BB%A3%E7%A0%81%20pip,%E4%B8%89%E3%80%81%E5%A6%82%E4%BD%95%E5%9C%A8Flask%E4%B8%AD%E6%B7%BB%E5%8A%A0%E5%9B%BE%E7%89%87%20%E5%9C%A8Flask%E4%B8%AD%E6%B7%BB%E5%8A%A0%E5%9B%BE%E7%89%87%E4%B8%8E%E6%B7%BB%E5%8A%A0%E6%A0%B7%E5%BC%8F%E8%A1%A8%E7%B1%BB%E4%BC%BC%EF%BC%8C%E6%88%91%E4%BB%AC%E9%9C%80%E8%A6%81%E5%B0%86%E5%9B%BE%E7%89%87%E6%96%87%E4%BB%B6%E6%94%BE%E5%9C%A8%E6%8C%87%E5%AE%9A%E7%9A%84%E9%9D%99%E6%80%81%E6%96%87%E4%BB%B6%E5%A4%B9%E4%B8%AD%EF%BC%8C%E5%B9%B6%E5%9C%A8HTML%E6%A8%A1%E6%9D%BF%E4%B8%AD%E5%BC%95%E7%94%A8%E5%AE%83%E4%BB%AC%E3%80%82%20%E4%BB%A5%E4%B8%8B%E6%98%AF%E8%AF%A6%E7%BB%86%E6%AD%A5%E9%AA%A4%EF%BC%9A%20%EF%BC%88%E4%B8%80%EF%BC%89%E5%88%9B%E5%BB%BA%E6%88%96%E7%A1%AE%E8%AE%A4%E9%9D%99%E6%80%81%E6%96%87%E4%BB%B6%E5%A4%B9%20%E7%A1%AE%E4%BF%9D%E6%88%91%E4%BB%AC%E7%9A%84Flask%E9%A1%B9%E7%9B%AE%E4%B8%AD%E6%9C%89%E4%B8%80%E4%B8%AA%E5%90%8D%E4%B8%BA%20static%20%E7%9A%84%E6%96%87%E4%BB%B6%E5%A4%B9%E3%80%82%20
"""

# app.py
from flask import Flask, render_template, request

app = Flask(__name__)

# 配置项（可选）
app.config['DEBUG'] = True  # 开启调试模式，这样代码变动后服务器会自动重启


# 路由和视图函数
@app.route('/')
def home():
    return render_template('index.html')  # 渲染模板文件


@app.route('/greet', methods=['GET', 'POST'])
def greet():
    if request.method == 'POST':
        name = request.form['name']  # 从表单中获取数据
        return f'Hello, {name}!'
    return render_template('greet.html')  # 渲染表单模板


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # 在所有网络接口上运行，监听5000端口