# RWKV-Light-API
A Lightweight and Extensible RWKV API for Inference

RWKV-Runner不可编辑？ChatRWKV功能太少？那么来尝试一下RWKV-Light-API吧。

提示：适合初步了解RWKV，希望能在它的基础上进行一定的扩展或研究的人群。

---
部署方法：
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
