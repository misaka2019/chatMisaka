# chatMisaka
使用chatGLM进行风格化微调

知乎链接 https://zhuanlan.zhihu.com/p/627942340

# 环境配置
~ pip install -r requirements.txt

# 使用教程
- 将chatGLM权重放到model文件夹下
- 更改eval.ipynb下的权重路径
- 一键全部运行eval.ipynb即可体验对话

# qq机器人部署教程
- 配置好go-cqhttp，相关教程https://blog.csdn.net/qq_51685718/article/details/127268883
- 运行main.py文件
- 登陆go-cqhttp即可体验qqbot。

# 不足与改进
- 模型现在仍然不会写代码，所以后续可以加入代码数据。
- 对知识的掌握不是很好，胡乱说的情况很多，所以可以增加相关知识进行训练。
- 模型现在只微调了三千条数据，改进空间很大。
- 生成方面，存在不断复读的情况，所以可以改进模型的采样方法。
