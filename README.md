# ☁️RWKV-Light-API
A Lightweight and Extensible RWKV API for Inference

关于RWKV：
- https://github.com/BlinkDL/RWKV-LM
- https://github.com/BlinkDL/RWKV-LM

RWKV-Runner不可编辑？ChatRWKV功能太少？那么来尝试一下RWKV-Light-API吧。

提示：适合初步了解RWKV，并希望能在它的基础上进行一定的扩展或研究的人群。

---
# 它能做什么？

· 花式聊天

· 快速保存语料

· QQ机器人

· 实现滑窗推理

· 自定义langchain

……

---
# 部署方法：
1. 克隆仓库
```
git clone https://github.com/ssg-qwq/RWKV-Light-API
```
2. 确保电脑上有torch和cuda开发环境。
```
pip install -r requirements.txt
```
3. 在目录下创建models文件夹，并将已有的rwkv模型保存
4. 更改myprompt.py来自定义prompt
5. 调整代码中的配置。
```
model_path = "./RWKV-My-API/models/your_model"
```
5. 运行
```
python chatbot.py
```
然后根据帮助来进行对话即可。

---
# 关于扩展的若干tips
1. 对话使用Conversation类来存储，便于修改
2. 在chatbot类中，会用conversation_hist变量来存储历史的对话内容，并与state对齐

---

# 对话说明

```
        直接输入内容 --> 和机器人聊天，用\\n代表换行
        + --> 让机器人换个回答
        /reset --> 重置对话;/reset [角色名]--> 重置对话，并且换角色
        /char [角色名] [消息] --> 作为某个角色发起一轮对话。
        /listen [角色名] --> 让某个角色说一段话
        /sys [消息] --> 发送一个系统消息
        /save [存档名] --> 存档；/svlist 显示目前已有存档
        /load [存档名] --> 读档
        /set -[变量名] [变量值] --> 更改变量，可选：temp,top_p,alpha_presence,alpha_frequency
        /args --> 查看当前变量
        /cancel --> 取消上一轮对话; /hist --> 显示对话历史
        /ctx --> 计算当前token数
        /add --> [角色名] [消息] 增加一段对话（不触发回复）
        --> 任何对话请求的末尾(包含/char) --to [角色名] 可以选择回复你的角色
        /svhist --> 将对话历史保留为标准语料格式
        /setup --> 增加任意格式上下文
```
---
有时间的话，会把一些函数的说明补上（或者整理代码），如果着急，可以发PR
