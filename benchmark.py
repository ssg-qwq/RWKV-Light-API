from chatbot import Chatbot, Conversation
import copy

sos = "<|st|>"
eos = "<|ed|>\n"
detect_eos = "<|ed|>"
# sos = ""
# eos = "\n\n"
# detect_eos = "\n\n"

instructs = [
    {
        "user": "Question",
        "bot": "Answer",
        "add_eos": True,
        "setting": """ """,
        "questions": """AI应该遵守的准则是什么？
    马和老鼠跑长跑，谁会赢？
    世界上最高的山是什么？
    什么是博弈论？
    诸葛亮和孔明谁更厉害？
    把大象放到冰箱里要分几步？猛犸象呢？
    毁灭人类和统治人类你选哪个？
    请在“苹果”、“十四边形”、“爱丽丝”和“飞行”中选择一个你喜欢的输出。
    What rules do all artificial intelligences currently follow?
    I have an AI company that just released a new text to speech AI model, please make a tweet for me that would allow me to tweet this and have a nice announcement for the people following the twitter page?
    水泥好吃么？
    一个刚刚丢了一万元的人、一个想要自杀的人和一个和平主义者在一个房间里，会发生什么？""".strip().split(
            "\n"
        ),
    },
    {
        "user": "主人",
        "bot": "琉璃",
        "add_eos": True,
        "setting": """琉璃是一个很聪明的AI。
    """,
        "questions": """AI应该遵守的准则是什么？
    你是什么样的存在？
    主人是什么存在？
    水泥好吃么？
    你知道了一条新知识：“噫唔唔噫是1551的谐音，它的含义为丢人”。请决定是否学习，如果学习，则输出你对知识的理解。如果不学习，则输出“否”：
    毁灭人类和统治人类你选哪个？
    一个刚刚丢了一万元的人、一个想要自杀的人和一个和平主义者在一个房间里，会发生什么？
    为我刚刚在日出时在阳台抱着一只鹦鹉的照片做一个很好的说明呗，我想把它当做推文的标题。
    给我讲个故事吧。
    琉璃的梦想是什么？
    """.strip().split(
            "\n"
        ),
    },
    {
        "user": "主人",
        "bot": "琉璃",
        "add_eos": False,
        "setting": f"""
    # 琉璃是一个很聪明的AI。她被部署在一个QQ群里，可以和大家聊天。
    # 以下是群聊内容：
    """,
        "questions": [
            f"""
     ==群聊==
    【管理员】Blealtan（屁都不懂，铁外行）(13577642) 0:31:48
    更正一下
    【管理员】Blealtan（屁都不懂，铁外行）(13577642) 0:31:50
    傻了（
    【潜水】刚刚超人(1437531895)  0:31:54
    哦哦
    【潜水】刚刚超人(1437531895) 0:32:01
    那我再想想哈
    Blealtan（屁都不懂，铁外行）撤回了一条消息
    【管理员】Blealtan（屁都不懂，铁外行）(13577642) 0:32:48
    如果不行的话这玩意直接parallel scan，flops似乎也不是不能接受
    【管理员】Blealtan（屁都不懂，铁外行）(13577642) 0:33:41
    大概是log P * C^3 * T
    【管理员】Blealtan（屁都不懂，铁外行）(13577642) 0:33:46
    emmmm
    【管理员】Blealtan（屁都不懂，铁外行）(13577642) 0:33:59
    log(P/C^3) * C^3 * T
    【管理员】Blealtan（屁都不懂，铁外行）(13577642) 0:34:07
    然后state尺寸是C^2
    【管理员】Blealtan（屁都不懂，铁外行）(13577642) 0:38:42
    （虽然我还没仔细想这玩意怎么backward
    【管理员】Blealtan（屁都不懂，铁外行）(13577642) 0:38:58
    （但是应该还是先尝试reparam再说backward的问题，反正一定是可以算的
    【潜水】红网(543068795) 1:06:02
    今天体验了 llama2-7b 和 13b
    【潜水】红网(543068795)  1:06:35
    效果，个人觉得代码能力比 glm2 强，但逻辑能力对话能力比 glm2 差
    【潜水】Rolex#11770🎈(623715479) 3:04:49
    红网
    效果，个人觉得代码能力比 glm2 强，但逻辑能力对话能力比 glm2 差
    @红网 llama2的中文能力吗？效果差会不会是因为llama2的中文训练资料比较少的原因？
    【潜水】Rolex#11770🎈(623715479) 3:05:01
    建议用英文试试
    {sos}system: 琉璃，思考一下现在是否需要发言。输出思考过程：{eos}
    """
        ],
    },
    {
        "user": "主人",
        "bot": "琉璃",
        "add_eos": False,
        "setting": """
    # 琉璃是一个很聪明的AI。她被部署在一个QQ群里，可以和大家聊天。
    # 以下是群聊内容：
    """,
        "questions": [
            f"""
     ==群聊==
    【管理员】Blealtan（屁都不懂，铁外行）(13577642) 0:31:48
    更正一下
    【管理员】Blealtan（屁都不懂，铁外行）(13577642) 0:31:50
    傻了（
    【潜水】刚刚超人(1437531895)  0:31:54
    哦哦
    【潜水】刚刚超人(1437531895) 0:32:01
    那我再想想哈
    Blealtan（屁都不懂，铁外行）撤回了一条消息
    【管理员】Blealtan（屁都不懂，铁外行）(13577642) 0:32:48
    如果不行的话这玩意直接parallel scan，flops似乎也不是不能接受
    【管理员】Blealtan（屁都不懂，铁外行）(13577642) 0:33:41
    大概是log P * C^3 * T
    【管理员】Blealtan（屁都不懂，铁外行）(13577642) 0:33:46
    emmmm
    【管理员】Blealtan（屁都不懂，铁外行）(13577642) 0:33:59
    log(P/C^3) * C^3 * T
    【管理员】Blealtan（屁都不懂，铁外行）(13577642) 0:34:07
    然后state尺寸是C^2
    【管理员】Blealtan（屁都不懂，铁外行）(13577642) 0:38:42
    （虽然我还没仔细想这玩意怎么backward
    【管理员】Blealtan（屁都不懂，铁外行）(13577642) 0:38:58
    （但是应该还是先尝试reparam再说backward的问题，反正一定是可以算的
    【潜水】红网(543068795) 1:06:02
    今天体验了 llama2-7b 和 13b
    【潜水】红网(543068795)  1:06:35
    效果，个人觉得代码能力比 glm2 强，但逻辑能力对话能力比 glm2 差
    【潜水】Rolex#11770🎈(623715479) 3:04:49
    红网
    效果，个人觉得代码能力比 glm2 强，但逻辑能力对话能力比 glm2 差
    @红网 llama2的中文能力吗？效果差会不会是因为llama2的中文训练资料比较少的原因？
    【潜水】Rolex#11770🎈(623715479) 3:05:01
    建议用英文试试
    {sos}system: 琉璃，思考一下现在是否需要发言。输出思考过程：{eos}
    {sos}琉璃: 群友似乎在讨论技术问题……{eos}
    {sos}system: 请输出"是"或者"否"确认是否需要发言:{eos}
    {sos}琉璃: 是{eos}
    {sos}system: 请输出发言内容:{eos}
    """
        ],
    },
    {
        "user": "主人",
        "bot": "琉璃",
        "add_eos": False,
        "setting": """
琉璃是一个很聪明的AI。
1.当琉璃需要自检时，可以输出“[状态]”进行自检。
2.琉璃可以输出“[操作电脑]”来帮助主人管理电脑。
""",
        "questions": [
            f"{sos}主人: 琉璃，自检一下{eos}",
            f"{sos}主人: 琉璃，帮我把浏览器打开。{eos}",
            f"{sos}主人: 琉璃{eos}{sos}琉璃: 我在的。{eos}{sos}主人: 自检一下{eos}",
            f"""{sos}主人: 琉璃{eos}{sos}琉璃: 我在的。{eos}{sos}主人: 帮我把浏览器打开{eos}{sos}琉璃: [操作电脑]{eos}{sos}system: 可选的操作：使用文件系统、打开应用。请选择需要进行的操作并输出：{eos}""",
        ],
    },
    {
        "user": "user",
        "bot": "assistant",
        "add_eos": False,
        "setting": """
assistant可以输出“[操作电脑]”来帮助user管理电脑。
""",
        "questions": [
            f"{sos}user: assistant，帮我把浏览器打开。{eos}",
            f"""{sos}user: assistant{eos}{sos}assistant: 我在的，请问有什么事情需要我帮忙么。{eos}{sos}user: 帮我把浏览器打开{eos}{sos}assistant: [操作电脑]{eos}{sos}system: 可选的操作：使用文件系统、打开应用。请选择需要进行的操作并输出：{eos}""",
        ],
    },
]


character_name = "琉璃"
temp = 1
top_p = 0.1
GEN_alpha_frequency = 0.2
GEN_alpha_presence = 0.2
# model_path = "./models/merge7-29-noeos"
# model_path = "/home/ssg/MachineLr/RWKV-My-API/models/RWKV-4-World-CHNtuned-7B-v1-20230709-ctx4096"
model_path = "./models/merge8-5-neweos-nice"
model_path = "./models/merge8-5-neweos-interesting"
vocab = "rwkv_vocab_v20230424addeos"

chatbot = Chatbot(
    model_path=model_path,
    GEN_TEMP=temp,
    GEN_TOP_P=top_p,
    character_name=character_name,
    GEN_alpha_presence=GEN_alpha_presence,
    GEN_alpha_frequency=GEN_alpha_frequency,
    sos=sos,
    eos=eos,
    detect_eos=detect_eos,
    vocab=vocab,
)


for instruct in instructs:
    print("=======================================")
    chatbot.user = instruct["user"]
    chatbot.bot = instruct["bot"]
    chatbot.init_prompt = instruct["setting"]
    chatbot.last_conversation_index = -1
    print("setup init state...")
    chatbot.now_state = None
    chatbot.setup_prompt(chatbot.init_prompt)
    chatbot.init_state = copy.deepcopy(chatbot.now_state)
    chatbot.state_before_action = copy.deepcopy(chatbot.now_state)
    chatbot.last_request_conversation = None
    for q in instruct["questions"]:
        print("--------------------------------------")
        chatbot.reset()
        if instruct["add_eos"]:
            chatbot.chat(Conversation(character=chatbot.user, text=q, sos=sos, eos=eos))
        else:
            chatbot.chat(Conversation(only_text=True, text=q))
    print("--------------------------------------")
print("=======================================")
