from flask import Flask, request
import api
from queue import Queue
import threading
from glm import predict

app = Flask(__name__)
request_queue = Queue()


def process_request():
    while True:
        # 从队列中取出请求消息
        data = request_queue.get()
        try:
            if data['message_type'] == 'group':  # 如果是群聊信息
                cq = data['raw_message']  # 获取带cq码的消息字段
                if cq[1:6] == 'CQ:at' and cq[10:20] == '#qq_id':  # 根据QQ号长度修改cq中的数字
                    gid = data['group_id']  # 获取群号
                    at_user_id = data['sender']['user_id']  # 获取@你的人是谁
                    at_message_text = cq[22:]  # 提取文本部分，根据QQ号长度修改cq中的数字
                    if len(at_message_text) > 500:
                        api.keyword("抱歉，字数过长", at_user_id, gid)
                    else:
                        message = predict(at_message_text)  # reply去调用chatgpt
                        print(message)
                        api.keyword(message, at_user_id, gid)

        except:
            pass

        # 标记请求消息已处理完成
        request_queue.task_done()


# 启动后台处理线程
thread = threading.Thread(target=process_request)
thread.daemon = True
thread.start()


@app.route('/', methods=["POST"])
def post_data():
    # 将请求消息加入队列，一个一个处理，主要防止画图请求过多，把GPU显存弄爆
    data = request.get_json()
    request_queue.put(data)
    return 'OK'


if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5701)
