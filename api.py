import requests


def keyword(message, uid, gid=None):
    if gid is None:  # 判断是否传入gid，区别群聊和私聊
        data = requests.get(url=f'http://127.0.0.1:5700/send_private_msg?user_id={uid}&message={message}')
        print(data)
        # 私聊回复requests网址，参数自行修改

    elif gid == 838039490:  # 群号，你想要自动回复的群
        if uid == 1635573329:  # 谁发的消息要回复
            requests.get(url='http://127.0.0.1:5700/send_group_msg?group_id=%s&message=%s' % (
                gid, '[CQ:at,qq=%s]我也爱你' % uid))
            # 群聊回复requests网址，参数自行修改
