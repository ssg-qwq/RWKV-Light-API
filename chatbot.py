import os, copy, types, gc, sys
from typing import Any
import numpy as np
import myprompt
from prompt_toolkit import prompt

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
os.environ[
    "RWKV_JIT_ON"
] = "1"  # '1' or '0', please use torch 1.13+ and benchmark speed
os.environ[
    "RWKV_CUDA_ON"
] = "0"  # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries
from rwkv.model import RWKV
from rwkv.utils import PIPELINE


class Conversation:
    def __init__(
        self,
        character: str = "",
        text: str = "",
        interface: str = ":",
        sos: str = "",
        eos: str = "\n\n",
        only_text=False,
    ) -> None:
        self.character = character
        self.text = text
        self.interface = interface
        self.only_text = only_text
        self.sos = sos
        self.eos = eos

    def __call__(self, add_space=True, *args: Any, **kwds: Any) -> Any:
        if not self.only_text:
            t = self.text.strip().replace("\r\n", "\n").replace("\n\n", "\n")
            if add_space:
                return f"{self.sos}{self.character}{self.interface} {t}{self.eos}"
            else:
                return f"{self.sos}{self.character}{self.interface}{t}{self.eos}"
        else:
            return self.text


class RWKV_Dynamic_Args:
    def __init__(
        self,
        CHUNK_LEN=256,
        CHAT_LEN_SHORT=4,
        CHAT_LEN_LONG=512,
        FREE_GEN_LEN=256,
        GEN_TEMP=1,
        GEN_TOP_P=0.85,
        GEN_alpha_presence=0.2,
        GEN_alpha_frequency=0.2,
    ):
        self.CHUNK_LEN = CHUNK_LEN
        self.CHAT_LEN_SHORT = CHAT_LEN_SHORT
        self.CHAT_LEN_LONG = CHAT_LEN_LONG
        self.FREE_GEN_LEN = FREE_GEN_LEN
        self.GEN_TEMP = GEN_TEMP
        self.GEN_TOP_P = GEN_TOP_P
        self.GEN_alpha_presence = GEN_alpha_presence
        self.GEN_alpha_frequency = GEN_alpha_frequency

    def to_string(self):
        for attr_name in dir(self):
            if not attr_name.startswith("__") and not attr_name == "to_string":
                attr_value = getattr(self, attr_name)
                print(f"{attr_name} : {attr_value}")


class Chatbot:
    """
    chatbot存为三部分:
    1.设定的state
    2.交互过程前的state，不含类langchain交互则为当前state
    3.交互过程前的text，不含类langchain交互则为当前text
    """

    def __init__(
        self,
        model_path="./models/RWKV-4-World-CHNtuned-3B-v1-20230625-ctx4096",  # 模型路径，需要写模型名并不要带.pth
        scroll_tokens=0,  # 为0的话则为RWKV推理模式，为其他数值则为GPT式，保留历史多少token（超出后，最开始的对话会被遗忘）
        load_history: str = None,  # 若读取对话存档（包括state、参数和prompt），则填入存档名（不含扩展名）
        strategy="cuda:1 bf16",  # RWKV Strategy
        prompt_path="./myprompt.py",  # 读取初始设定
        character_name="琉璃",
        states_path="./states",  # 存state的路径
        history_path="./history",  # 对话存档的路径
        system="System",  # 系统角色
        sos="",  # Start Of Sentence 字符，RWKV默认为''
        eos="\n\n",  # End Of Sentence 字符，RWKV默认为'\n\n'，用来自动生成格式
        detect_eos="\n\n",  # 检测用的eos，RWKV默认为'\n\n'，用来判断什么时候停止，如果没有特殊需求，eos和detect_eos应该是一致的
        debug_mode=False,
        CHUNK_LEN=256,
        CHAT_LEN_SHORT=4,
        CHAT_LEN_LONG=512,
        FREE_GEN_LEN=256,
        GEN_TEMP=1,
        GEN_TOP_P=0.7,
        GEN_alpha_presence=0.2,
        GEN_alpha_frequency=0.2,
        vocab="rwkv_vocab_v20230424",
    ) -> None:
        self.prompt_path = prompt_path
        self.states_path = states_path
        self.history_path = history_path

        print(f"Loading model - {model_path} - {strategy}")

        self.model = RWKV(model=model_path, strategy=strategy)
        self.pipeline = PIPELINE(self.model, vocab)  # world模型tokenizer管线

        self.init_state = None  # 存储设定和说明书的state
        self.state_before_action = None  # 交互前的state或上一轮的state
        self.now_state = None  # 当前state

        self.conversation_hist = []  # 对话记录
        self.last_conversation_index = 0

        # 格式相关
        (
            self.user,
            self.bot,
            self.interface,
            self.init_prompt,
        ) = self.ChatRWKV_backend_load_prompt(prompt_path, character_name)
        self.system = system
        self.sos = sos
        self.eos = eos
        self.detect_eos = detect_eos

        self.debug_mode = debug_mode

        self.scroll_tokens = scroll_tokens

        self.debug_count = 0

        # 可以动态调节的参数 初始化
        self.dynamic_settings = RWKV_Dynamic_Args(
            CHUNK_LEN=CHUNK_LEN,
            CHAT_LEN_SHORT=CHAT_LEN_SHORT,
            CHAT_LEN_LONG=CHAT_LEN_LONG,
            FREE_GEN_LEN=FREE_GEN_LEN,
            GEN_TEMP=GEN_TEMP,
            GEN_TOP_P=GEN_TOP_P,
            GEN_alpha_presence=GEN_alpha_presence,
            GEN_alpha_frequency=GEN_alpha_frequency,
        )
        self.END_OF_TEXT = 0
        self.END_OF_LINE = 187
        self.END_OF_LINE_DOUBLE = 535
        help_msg = """指令:
        直接输入内容 --> 和机器人聊天，用\\n代表换行
        + --> 让机器人换个回答
        /reset --> 重置对话;/reset [角色名]--> 重置对话，并且换角色
        /char [角色名] [消息] 作为某个角色发起一轮对话。
        /listen [角色名] 让某个角色说一段话
        /sys [消息] 发送一个系统消息
        /save [存档名] 存档；/svlist 显示目前已有存档
        /load [存档名] 读档
        /set -[变量名] [变量值] 更改变量，可选：temp,top_p,alpha_presence,alpha_frequency
        /args 查看当前变量
        /cancel 取消上一轮对话; /hist 显示对话历史
        /ctx 计算当前token数
        /add [角色名] [消息] 增加一段对话（不触发回复）
        任何对话请求的末尾(包含/char) --to [角色名] 可以选择回复你的角色
        """
        print(help_msg)
        if load_history is not None:
            self.load_history(load_history)
        else:
            # 加载初始prompt
            print("setup init state...")
            self.setup_prompt(self.init_prompt)
            self.init_state = copy.deepcopy(self.now_state)
            self.state_before_action = copy.deepcopy(self.now_state)
        self.callback_res = None
        self.last_request_conversation = None
        gc.collect()

    def conversation2text(self, convesations: list[Conversation]):
        """将Conversation列表转为prompt字符串"""
        s = ""
        for convesation in convesations:
            s += convesation()
        return s

    def save_state(self, state, state_name: str = "setting.state", to_path=None):
        """存state，state是RWKV的循环向量，与GPT的聊天历史等价"""
        if not os.path.exists(self.states_path):
            os.mkdir(self.states_path)
        path = (
            os.path.join(self.states_path, state_name)
            if to_path is None
            else os.path.join(to_path, state_name)
        )
        if self.debug_mode:
            print(f"save state from {path}")
        torch.save(state, path)

    def load_state(self, state_name, from_path=None):
        """读state"""
        path = (
            os.path.join(self.states_path, state_name)
            if from_path is None
            else os.path.join(from_path, state_name)
        )
        state = torch.load(path)
        if self.debug_mode:
            print(f"load state from: {path}")
        return state

    def save_history(self, hist_name="存档1"):
        """就像游戏存档一样把当前的情况全存下来"""
        path = os.path.join(self.history_path, hist_name + ".svpt")
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        svpt = {
            "init_state": self.init_state,  # 存初始state
            "state_before_action": self.state_before_action,  # 存之前state
            "now_state": self.now_state,  # 存当前state
            "dynamic_settings": self.dynamic_settings,  # 存动态参数
            "conversation_hist": self.conversation_hist,  # 存聊天记录
            "last_conversation_index": self.last_conversation_index,
            "init_prompt": self.init_prompt,
        }
        torch.save(svpt, path)
        print(f"save {hist_name} at {path}")

    def load_history(self, hist_name):
        """读档"""
        path = os.path.join(self.history_path, hist_name + ".svpt")
        try:
            svpt = torch.load(path)
        except:
            print("找不到存档")
            return
        self.init_state = svpt["init_state"]
        self.state_before_action = svpt["state_before_action"]
        self.now_state = svpt["now_state"]
        self.dynamic_settings = svpt["dynamic_settings"]
        self.conversation_hist = svpt["conversation_hist"]
        self.last_conversation_index = svpt["last_conversation_index"]
        self.init_prompt = svpt["init_prompt"]
        # 显示
        if self.debug_mode:
            print(self.conversation2text(self.conversation_hist))
        else:
            print(
                self.conversation2text(self.conversation_hist)
                .replace(self.sos, "")
                .replace(self.eos, "\n")
            )
        print(f"load {hist_name} from {path}")
        print(self.init_prompt.replace("\r\n", "\n").replace("\n\n", "\n"))
        for c in self.conversation_hist:
            print(c().replace("\r\n", "\n").replace("\n\n", "\n"))

    def setup_prompt(self, prompt: str, add_conversation=False):
        """不触发回应，仅在当前state下加载一段新的prompt

        Args:
            add_conversation (bool, optional): 是否把这段加入历史对话. Defaults to False.
        """
        _, self.now_state = self.RWKV_infer(self.encode(prompt), self.now_state)
        # 显示
        if self.debug_mode:
            print(prompt)
        else:
            print(prompt.replace(self.sos, "").replace(self.eos, "\n"))
        if add_conversation:
            self.conversation_hist.append(Conversation(only_text=True, text=prompt))

    def chat(
        self,
        conversation: Conversation,
        mark_before_action=True,
        to_char: str = None,
        avoid_talking2oneself=True,
        response_callback=None,
        *callback_args,
        **callback_kwargs,
    ):
        """_summary_

        Args:
            conversation (Conversation): 对话对象
            mark_before_action (bool, optional): 是否记录，记录的话，重新生成会从这里开始. Defaults to True.
            to_char (str, optional): 希望哪个角色来回复你，不填则为默认. Defaults to None.
            avoid_talking2oneself (bool, optional): 如果bot输出了{user}{interface}则强制结束，并修改结尾为eos. Defaults to True.
            response_callback (__callable__, optional): 回调函数. Defaults to None.

        Returns:
            str : response_str
        """
        self.debug_count += 1
        if mark_before_action:
            self.state_before_action = self.now_state
        if self.debug_mode:
            print(conversation().replace("\n", "\\n"))
        else:
            print(conversation().strip("\n"))
        # else:
        #     print(conversation().replace(self.sos,'').replace(self.eos,'\n'))
        if to_char is None:
            to_char = self.bot
        out, state = self.RWKV_infer(
            self.encode(f"{conversation()}{self.sos}{to_char}{self.interface}"),
            self.now_state,
            newline_adj=-999999999,
        )
        # 采样
        tokens = []
        begin = 0
        out_last = begin
        x_temp = self.dynamic_settings.GEN_TEMP
        x_top_p = self.dynamic_settings.GEN_TOP_P
        if self.debug_mode:
            print(f"x_temp:{x_temp},top_p:{x_top_p}")
        print(f"{to_char}{self.interface}", end="", flush=True)
        occurrence = {}

        if x_temp <= 0.2:
            x_temp = 0.2
        if x_temp >= 5:
            x_temp = 5
        if x_top_p <= 0:
            x_top_p = 0
        for i in range(999):
            if i <= 0:
                newline_adj = -999999999
            elif i <= self.dynamic_settings.CHAT_LEN_SHORT:
                newline_adj = (i - self.dynamic_settings.CHAT_LEN_SHORT) / 10
            elif i <= self.dynamic_settings.CHAT_LEN_LONG:
                newline_adj = 0
            else:
                newline_adj = min(
                    3, (i - self.dynamic_settings.CHAT_LEN_LONG) * 0.25
                )  # MUST END THE GENERATION

            for n in occurrence:
                out[n] -= (
                    self.dynamic_settings.GEN_alpha_presence
                    + occurrence[n] * self.dynamic_settings.GEN_alpha_frequency
                )
            token = self.pipeline.sample_logits(
                out,
                temperature=x_temp,
                top_p=x_top_p,
            )
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

            out, state = self.RWKV_infer([token], state, newline_adj=newline_adj)
            tokens += [token]
            out[self.END_OF_TEXT] = -999999999  # disable <|endoftext|>

            xxx = self.pipeline.decode(tokens[out_last:])
            if "\ufffd" not in xxx:  # avoid utf-8 display issues
                if self.debug_mode:
                    print(
                        xxx.replace("\r\n", "\\r\\n\n").replace(
                            detect_eos, f"{self.detect_eos}\n"
                        ),
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        xxx.replace("\r\n", "\n").replace(self.detect_eos, "\n"),
                        end="",
                        flush=True,
                    )
                out_last = begin + i + 1

            uc = f"{self.user}{self.interface}"
            send_msg = self.pipeline.decode(tokens[begin:])
            if self.detect_eos in send_msg:
                send_msg = send_msg.strip()
                break
            elif avoid_talking2oneself and send_msg[len(uc) :] == uc:
                print("bot tried to talk in oneself")
                send_msg = send_msg.replace(uc, "\n").strip()
                break
        # 处理状态
        response_conversation = Conversation(
            to_char, send_msg.replace(self.detect_eos, ""), sos=self.sos, eos=self.eos
        )
        if mark_before_action:
            self.last_conversation_index = len(self.conversation_hist)
        if conversation() != "":
            self.conversation_hist.append(conversation)
        self.conversation_hist.append(response_conversation)
        while self.scroll_tokens != 0 and self.calc_ctx() > self.scroll_tokens:
            self.forget()
        self.now_state = state
        if response_callback is not None:
            self.callback_res = response_callback(
                self, send_msg, *callback_args, **callback_kwargs
            )

        return send_msg.replace(self.detect_eos, "")

    def process_reaction_end(self, insert_prompt):
        """
        回退到上次对话请求前并插入prompt
        使用样例: process_reaction_end(user_conversation()+"<系统操作过程已折叠>\\n\\n"+bot_conversation()")
        """
        self.back2_last_recation()
        if insert_prompt != "":
            _, self.now_state = self.RWKV_infer(
                self.encode(insert_prompt), self.state_before_action
            )
            self.conversation_hist.append(
                Conversation(only_text=True, text=insert_prompt)
            )
        self.last_conversation_index = len(self.conversation_hist)

    def forget(self):
        """删除最开始的那条消息"""
        self.conversation_hist.pop(0)
        self.last_conversation_index -= 1
        s = ""
        for i in range(self.last_conversation_index):
            s += self.conversation_hist[i]()
        ls = s
        for c in self.conversation_hist[self.last_conversation_index :]:
            s += c()

        _, self.state_before_action = self.RWKV_infer(self.encode(ls), self.init_state)
        _, self.now_state = self.RWKV_infer(self.encode(s), self.init_state)

    def calc_ctx(self):
        """计算当前token"""
        s = self.init_prompt
        for conv in self.conversation_hist:
            s += conv()
        tokens = self.encode(s)
        return len(tokens)

    def reset(self, change_character: str = None):
        """重置bot"""
        print("bot已重置")
        if change_character is None:
            self.now_state = copy.deepcopy(self.init_state)
            self.conversation_hist.clear()
            self.last_conversation_index = -1
            print(self.init_prompt)
        else:
            # 格式相关
            self.conversation_hist.clear()
            (
                self.user,
                self.bot,
                self.interface,
                self.init_prompt,
            ) = self.ChatRWKV_backend_load_prompt(self.prompt_path, change_character)
            self.last_conversation_index = -1

            print("setup init state...")
            self.now_state = None
            self.setup_prompt(self.init_prompt)
            self.init_state = copy.deepcopy(self.now_state)
            self.state_before_action = copy.deepcopy(self.now_state)
            self.last_request_conversation = None

    def regenerate(self, to_char=None):
        self.back2_last_recation()
        self.chat(self.last_request_conversation, to_char=to_char)

    def back2_last_recation(self):
        self.now_state = self.state_before_action
        if len(self.conversation_hist) > self.last_conversation_index:
            self.last_request_conversation = self.conversation_hist[
                self.last_conversation_index
            ]
            self.conversation_hist = self.conversation_hist[
                : self.last_conversation_index
            ]
        elif self.last_conversation_index == 0:
            self.conversation_hist.clear()
            self.last_request_conversation = None
        if self.debug_mode:
            if len(self.conversation_hist) == 0:
                print("back to: begining")
            else:
                print(
                    "back to: ",
                    self.conversation_hist[-1]()
                    .replace("\r\n", "\n")
                    .replace(detect_eos, "\n"),
                )

    def add_fake_conversation(self, conversation: Conversation):
        """增加一条虚假的对话信息"""
        self.setup_prompt(conversation())
        self.conversation_hist.append(conversation)

    def RWKV_infer(self, tokens, state, newline_adj=0):
        """_summary_

        Args:
            tokens (_type_): 经过pipeline tokenize之后的输入
            state (_type_): 模型状态

        Returns:
            _type_: out,state
        """
        model_state = copy.deepcopy(state)
        tokens = [int(x) for x in tokens]
        # print(f'### model ###\n{tokens}\n[{pipeline.decode(model_tokens)}]')
        while len(tokens) > 0:
            out, model_state = self.model.forward(
                tokens[: self.dynamic_settings.CHUNK_LEN], model_state
            )
            tokens = tokens[self.dynamic_settings.CHUNK_LEN :]
        out[self.END_OF_LINE] += newline_adj  # adjust \n probability
        return out, model_state

    def encode(self, text: str):
        """将文本转化为RWKV World词向量"""
        return self.ChatRWKV_backend_fix_tokens(self.pipeline.encode(text))

    def ChatRWKV_backend_fix_tokens(self, tokens):
        """沿用ChatRWKV的fix_tokens方法，修复换行"""
        if len(tokens) > 0 and tokens[-1] == self.END_OF_LINE_DOUBLE:
            tokens = tokens[:-1] + [self.END_OF_LINE, self.END_OF_LINE]
        return tokens

    def ChatRWKV_backend_load_prompt(self, PROMPT_FILE, character_name):
        """沿用ChatRWKV的load_prompt方法，读取对话格式"""
        variables = {}
        with open(PROMPT_FILE, "rb") as file:
            exec(compile(file.read(), PROMPT_FILE, "exec"), variables)
        user, bot, interface, init_prompt = (
            variables[f"user"],
            variables[f"bot_{character_name}"],
            variables[f"interface"],
            variables[f"init_prompt_{character_name}"],
        )
        init_prompt = init_prompt.strip().split("\n")
        for c in range(len(init_prompt)):
            init_prompt[c] = init_prompt[c].strip().strip("\u3000").strip("\r")
        init_prompt = "\n" + ("\n".join(init_prompt)).strip() + "\n\n"
        return user, bot, interface, init_prompt


# test script
if __name__ == "__main__":
    # model_path = "./models/merge7-22-noeos-nice"
    # model_path = "./models/merge7-22-noeos-interesting"
    # model_path = "./models/merge7-25-noeos-nice"
    # model_path = "./models/merge7-29-noeos-interesting"
    # model_path = "./models/mianwa"
    # model_path = "./models/merge7-29-noeos-nice"
    model_path = "./models/merge7-29-noeos"
    # model_path = "./models/merge8-5-neweos-interesting"
    # model_path = "./models/merge8-5-neweos"
    # model_path = "/home/ssg/MachineLr/RWKV-My-API/models/RWKV-4-World-CHNtuned-7B-v1-20230709-ctx4096"
    scroll_tokens = 0
    temp = 1
    top_p = 0.7
    GEN_alpha_frequency = 0.2
    GEN_alpha_presence = 0.2
    character_name = "琉璃"
    character_name = "assistant"
    # sos = "<|st|>"
    # eos = "<|ed|>\n"
    # detect_eos = "<|ed|>"
    sos = ""
    eos = "\n\n"
    detect_eos = "\n\n"
    # vocab="rwkv_vocab_v20230424"
    vocab = "rwkv_vocab_v20230424addeos"
    # character_name='测试'
    chatbot = Chatbot(
        model_path=model_path,
        scroll_tokens=scroll_tokens,
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
    to = None
    while True:
        msg = prompt(">")
        msg = msg.replace("\\n", "\n").strip()
        if msg[:6].lower() == "/char ":
            char = msg[6:].split(" ")[0]
            real_msg_s = msg[6:].split(" ")[1:]
            real_msg = ""
            for s in real_msg_s:
                real_msg += s + " "
            if "--to " in real_msg:
                to = real_msg.split("--to ")[1]
                real_msg = real_msg.split("--to ")[0]
            else:
                to = None
            real_msg = real_msg.strip()
            conversation = Conversation(character=char, text=real_msg, sos=sos, eos=eos)
            chatbot.chat(conversation, to_char=to)
        elif msg.lower() == "/listen":
            chatbot.chat(
                Conversation(only_text=True, text=""), to_char=to, sos=sos, eos=eos
            )
        elif msg[:8].lower() == "/listen ":
            to = msg[8:].strip()
            chatbot.chat(
                Conversation(only_text=True, text=""), to_char=to, sos=sos, eos=eos
            )
        elif msg[:5].lower() == "/sys ":
            real_msg = msg[5:].strip()
            conversation = Conversation(
                character=chatbot.system, text=real_msg, sos=sos, eos=eos
            )
            chatbot.chat(conversation)
        elif msg[:5].lower() == "/add ":
            char = msg[5:].split(" ")[0]
            real_msg_s = msg[5:].split(" ")[1:]
            real_msg = ""
            for s in real_msg_s:
                real_msg += s + " "
            real_msg = real_msg.strip()
            conversation = Conversation(character=char, text=real_msg, sos=sos, eos=eos)
            chatbot.add_fake_conversation(conversation)
        elif msg.lower() == "+":
            if len(chatbot.conversation_hist) != 0:
                chatbot.regenerate(to)
            else:
                print("你还没有历史消息怎么重新生成啊...")
        elif msg.lower() == "/reset":
            chatbot.reset()
        elif msg[:7].lower() == "/reset ":
            char = msg[7:].strip()
            try:
                chatbot.reset(char)
            except:
                print("找不到这个角色")
        elif msg[:6].lower() == "/save ":
            name = msg[6:].strip()
            chatbot.save_history(name)
        elif msg[:6].lower() == "/load ":
            name = msg[6:].strip()
            chatbot.load_history(name)
        elif msg[:5].lower() == "/set ":
            if "-temp=" in msg:
                arg = float(msg.split("-temp=")[1].split(" ")[0])
                chatbot.dynamic_settings.GEN_TEMP = arg
                print(f"temp={arg}")
            if "-top_p=" in msg:
                arg = float(msg.split("-top_p=")[1].split(" ")[0])
                chatbot.dynamic_settings.GEN_TOP_P = arg
                print(f"top_p={arg}")
            if "-alpha_presence=" in msg:
                arg = float(msg.split("-alpha_presence=")[1].split(" ")[0])
                chatbot.dynamic_settings.GEN_alpha_presence = arg
                print(f"alpha_presence={arg}")
            if "-alpha_frequency=" in msg:
                arg = float(msg.split("-alpha_frequency=")[1].split(" ")[0])
                chatbot.dynamic_settings.GEN_alpha_frequency = arg
                print(f"alpha_frequency={arg}")
        elif msg.lower() == "/svlist":
            dirs = os.listdir(chatbot.history_path)
            svpts = [d for d in dirs if d.endswith(".svpt")]
            print(svpts)
        elif msg.lower() == "/cancel":
            chatbot.back2_last_recation()
            print("对话已撤回")
        elif msg.lower() == "/ctx":
            print(chatbot.calc_ctx())
        elif msg.lower() == "/hist":
            for c in chatbot.conversation_hist:
                print(c().replace("\r\n", "\n").replace("\n\n", "\n"))
        elif msg.lower() == "/args":
            print(chatbot.dynamic_settings.to_string())
        else:
            if "--to " in msg:
                to = msg.split("--to ")[1]
                msg = msg.split("--to ")[0]
            else:
                to = None
            conversation = Conversation(
                character=chatbot.user, text=msg, sos=sos, eos=eos
            )
            chatbot.chat(conversation, to_char=to)
