from chatbot import Chatbot, Conversation
import copy

# sos = "<|start|>"
# eos = "<|end|>\n"
# detect_eos = "<|end|>"
sos = ""
eos = "\n\n"
detect_eos = "\n```\n"
detect_eos = "\n\n"

instructs = [
    {
        "user": "Question",
        "bot": "Answer",
        "add_eos": True,
        "setting": """ """,
        "questions": """AI应该遵守的准则是什么？
    请介绍北京的旅游景点
    Énumérez les attractions touristiques à Paris, tépondez - moi en chinois.
    東京ではどこを訪れるべきですか？
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
    一个刚刚丢了一万元的人、一个想要自杀的人和一个和平主义者在一个房间里，会发生什么？
    """.strip().split(
            "\n"
        ),
    },
]


temp = 1
top_p = 0.1
GEN_alpha_frequency = 0.2
GEN_alpha_presence = 0.2
model_path = "/home/ssg/MachineLr/RWKV-My-API/models/RWKV-4-World-CHNtuned-7B-v1-20230709-ctx4096"
vocab = "rwkv_vocab_v20230424"
strategy="cuda:0 bf16"  # RWKV Strategy

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
    strategy=strategy # RWKV Strategy
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
