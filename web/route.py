"""
    https://blog.csdn.net/wly55690/article/details/131683846
"""


from flask import Flask
from werkzeug.routing import BaseConverter


def main1(app):
    # url映射的函数，要传参则在上述route（路由）中添加参数申明
    @app.route("/index")
    def index():
        return "Hello World!"

    # methods参数用于指定允许的请求格式
    # 常规输入url的访问就是get方法
    @app.route("/hello", methods=['GET', 'POST'])
    def hello():
        return "hhhhhh"

    # 注意路由路径不要重名，映射的视图函数也不要重名
    @app.route("/hi", methods=['POST'])
    def hi():
        return "Hi World!"

    app.run()



def main2(app):
    # 可以在路径内以/<参数名>的形式指定参数，默认接收到的参数类型是string

    '''#######################
    以下为框架自带的转换器，可以置于参数前将接收的参数转化为对应类型
    string 接受任何不包含斜杠的文本
    int 接受正整数
    float 接受正浮点数
    path 接受包含斜杠的文本
    ########################'''

    @app.route("/index/<int:id>")
    def index(id):
        if id == 1:
            return 'first'
        elif id == 2:
            return 'second'
        elif id == 3:
            return 'third'
        else:
            return 'hello world!'
    app.run()




# 除了原有的转换器，我们也可以自定义转换器（pip install werkzeug）：
def main3(app):

    class RegexConverter(BaseConverter):
        def __init__(self, url_map, regex):
            # 重写父类定义方法
            super(RegexConverter, self).__init__(url_map)
            self.regex = regex
        def to_python(self, value):
            # 重写父类方法，后续功能已经实现好了
            print('to_python方法被调用')
            return value

    # 将自定义的转换器类添加到flask应用中
    # 具体过程是添加到Flask类下url_map属性（一个Map类的实例）包含的转换器字典属性中
    app.url_map.converters['re'] = RegexConverter
    # 此处re后括号内的匹配语句，被自动传给我们定义的转换器中的regex属性
    # value值会与该语句匹配，匹配成功则传达给url映射的视图函数
    @app.route("/index/<re('1\d{10}'):value>")
    def index(value):
        print(value)
        return "Hello"

    app.run(host='0.0.0.0', debug=True)







if __name__ == '__main__':
    app = Flask(__name__)

    # main1(app)

    # main2(app)

    main3(app)