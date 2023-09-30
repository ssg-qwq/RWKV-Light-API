import os, copy, types, gc, sys
from typing import Any
import numpy as np
import myprompt
import time
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


chat_sos_tokens = [65530, 65532]
chat_eos_tokens = [65535, 11]
stop_add_token = [11]
system_sos_tokens = [65530, 65531]
system_eos_tokens = [65535, 11]

think_sos_tokens = [65530, 65533]
think_eos_tokens = [65535, 11]
detect_eos_token = [65535]


class Conversation:
    def __init__(
        self,
        character: str = "",
        text: str = "",
        interface: str = ":",
        sos_tokens=[],
        eos_tokens=[],
        sos: str = "",
        eos: str = "",
        only_text=False,
    ) -> None:
        self.character = character
        self.text = text
        self.interface = interface
        self.only_text = only_text
        self.sos_tokens = sos_tokens
        self.eos_tokens = eos_tokens
        self.sos = sos
        self.eos = eos

    def __call__(self, add_space=True, *args: Any, **kwds: Any) -> Any:
        if not self.only_text:
            t = self.text.strip()
            if add_space:
                return f"{self.sos}{self.character}{self.interface} {t}{self.eos}"
            else:
                return f"{self.sos}{self.character}{self.interface}{t}{self.eos}"
        else:
            return self.text

    def to_tokens(self, encoding_func: callable, add_space=True):
        # print(self.sos_tokens + encoding_func(self(add_space)) + self.eos_tokens)
        return self.sos_tokens + encoding_func(self(add_space)) + self.eos_tokens


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
        sos_tokens=[],  # Start Of Sentence 字符，RWKV默认为''
        eos_tokens=[261],  # End Of Sentence 字符，RWKV默认为'\n\n'，用来自动生成格式
        detect_eos_token=[
            261,
            263,
        ],  # 检测用的eos，RWKV默认为'\n\n'，用来判断什么时候停止，如果没有特殊需求，eos和detect_eos应该是一致的
        stop_add_token=[],
        debug_mode=False,
        CHUNK_LEN=256,
        CHAT_LEN_SHORT=4,
        CHAT_LEN_LONG=512,
        FREE_GEN_LEN=256,
        GEN_TEMP=1,
        GEN_TOP_P=0.7,
        GEN_alpha_presence=0.2,
        GEN_alpha_frequency=0.2,
        think_sos_tokens=None,
        think_eos_tokens=None,
        think_desire=5,
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
        self.sos_tokens = sos_tokens
        self.eos_tokens = eos_tokens
        self.stop_add_token = stop_add_token
        self.detect_eos_token = detect_eos_token
        self.think_sos_tokens = (
            [65530, 65533] if think_sos_tokens is None else think_sos_tokens
        )
        self.think_eos_tokens = (
            [65535, 11] if think_eos_tokens is None else think_eos_tokens
        )
        self.think_desire=think_desire

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
        print(self.conversation2text(self.conversation_hist))
        print(f"load {hist_name} from {path}")
        print(self.init_prompt.strip())
        for c in self.conversation_hist:
            print(c().strip())

    def setup_prompt(
        self, prompt: str, add_conversation=False, sos_tokens=[], eos_tokens=[]
    ):
        """不触发回应，仅在当前state下加载一段新的prompt

        Args:
            add_conversation (bool, optional): 是否把这段加入历史对话. Defaults to False.
        """
        self.add_fake_conversation(
            Conversation(
                only_text=True,
                text=prompt,
                sos_tokens=sos_tokens,
                eos_tokens=eos_tokens,
            ),
            add_to_hist=add_conversation,
        )

    def setup_conversations(self, conversations, add_to_hist=True):
        for conversation in conversations:
            self.add_fake_conversation(conversation, add_to_hist)

    def chat(
        self,
        conversation: Conversation,
        mark_before_action=True,
        to_char: str = None,
        response_callback=None,
        expect_sp_token=None,
        allow_think=True,
        *callback_args,
        **callback_kwargs,
    ):
        """_summary_

        Args:
            conversation (Conversation): 对话对象
            mark_before_action (bool, optional): 是否记录，记录的话，重新生成会从这里开始. Defaults to True.
            to_char (str, optional): 希望哪个角色来回复你，不填则为默认. Defaults to None.
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
        if expect_sp_token is None:
            now_sos_tokens = self.sos_tokens
        else:
            now_sos_tokens = expect_sp_token
        start_str = f"{to_char}{self.interface}"

        if mark_before_action:
            self.last_conversation_index = len(self.conversation_hist)
        if conversation() != "":
            self.conversation_hist.append(conversation)

        if allow_think and expect_sp_token is None:
            self.try_think(to_char,self.think_desire)

        out, state = self.RWKV_infer(
            conversation.to_tokens(self.encode)
            + now_sos_tokens
            + self.encode(start_str),
            self.now_state,
        )

        # 采样
        tokens = []
        begin = 0
        x_temp = self.dynamic_settings.GEN_TEMP
        x_top_p = self.dynamic_settings.GEN_TOP_P
        if self.debug_mode:
            print(f"x_temp:{x_temp},top_p:{x_top_p}")
        print(start_str, end="", flush=True)
        occurrence = {}

        uc = self.sos_tokens + self.encode(f"{self.user}{self.interface}")
        luc = len(uc)
        for i in range(999):
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

            out, state = self.RWKV_infer([token], state)
            tokens += [token]
            # avoid to talk with self
            if tokens[-luc:] == uc:
                tokens = tokens[:-luc]
                char_count = len(f"{self.user}{self.interface}")
                for c in range(char_count):
                    print("\b \b" * char_count, end="")
                    sys.stdout.flush()
                break
            if token in self.detect_eos_token:
                _, state = self.RWKV_infer(self.stop_add_token, state)
                tokens.pop()
                break

            out[self.END_OF_TEXT] = -999999999  # disable <|endoftext|>

            xxx = self.pipeline.decode([token])

            print(xxx, end="")
            sys.stdout.flush()

        send_msg = self.pipeline.decode(tokens[begin:])

        # 处理状态
        response_conversation = Conversation(
            to_char,
            send_msg,
            sos_tokens=self.sos_tokens,
            eos_tokens=self.eos_tokens,
            sos="",
            eos="",
        )
        self.conversation_hist.append(response_conversation)
        while self.scroll_tokens != 0 and self.calc_ctx() > self.scroll_tokens:
            self.forget()
        self.now_state = state
        if response_callback is not None:
            self.callback_res = response_callback(
                self, send_msg, *callback_args, **callback_kwargs
            )

        return send_msg

    def process_reaction_end(
        self, insert_conversations=[], sos_tokens=[], eos_tokens=[]
    ):
        """
        回退到上次对话请求前并插入prompt
        使用样例: process_reaction_end(user_conversation()+"<系统操作过程已折叠>\\n\\n"+bot_conversation()")
        """
        self.back2_last_recation()
        if len(insert_conversations) != 0:
            s = self.state_before_action
            for c in insert_conversations:
                self.conversation_hist.append(c)
                _, s = self.RWKV_infer(c.to_tokens(self.encode), s)
            self.now_state = s
        self.last_conversation_index = len(self.conversation_hist)

    def forget(self):
        """删除最开始的那条消息"""
        self.conversation_hist.pop(0)
        self.last_conversation_index -= 1
        s = []
        for i in range(self.last_conversation_index):
            s += self.conversation_hist[i].to_tokens(self.encode)
        ls = s
        for c in self.conversation_hist[self.last_conversation_index :]:
            s += c.to_tokens(self.encode)

        _, self.state_before_action = self.RWKV_infer(ls, self.init_state)
        _, self.now_state = self.RWKV_infer(s, self.init_state)

    def calc_ctx(self):
        """计算当前token"""
        s = self.encode(self.init_prompt)
        for conv in self.conversation_hist:
            s += conv.to_tokens(self.encode)
        return len(s)

    # 估计要改一下
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
                print("back to: ", self.conversation_hist[-1]())

    def add_fake_conversation(self, conversation: Conversation, add_to_hist=True):
        """增加一条虚假的对话信息"""
        _, self.now_state = self.RWKV_infer(
            conversation.to_tokens(self.encode), self.now_state
        )
        print(conversation())
        if add_to_hist:
            self.conversation_hist.append(conversation)

    def RWKV_infer(self, tokens, state):
        """_summary_

        Args:
            tokens (_type_): 经过pipeline tokenize之后的输入
            state (_type_): 模型状态

        Returns:
            _type_: out,state
        """
        out = None
        model_state = copy.deepcopy(state)
        tokens = [int(x) for x in tokens]
        # print(f'### model ###\n{tokens}\n[{pipeline.decode(model_tokens)}]')
        while len(tokens) > 0:
            out, model_state = self.model.forward(
                tokens[: self.dynamic_settings.CHUNK_LEN], model_state
            )
            tokens = tokens[self.dynamic_settings.CHUNK_LEN :]
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

    def change_converlist_eos(self, conver_list, new_sos, new_eos):
        res = []
        for c in conver_list:
            res.append(
                Conversation(
                    sos=new_sos,
                    eos=new_eos,
                    text=c.text,
                    only_text=c.only_text,
                    character=c.character,
                )
            )
        return res

    def try_think(self, to_char, desire):
        already_think = False

        bc = self.think_sos_tokens + self.pipeline.encode(
            f"({self.bot}{self.interface}"
        )

        desire, think_state,think_out = self.estimate_desire(bc, chat_token_decay=desire)

        if desire:
            print(f"({self.bot}{self.interface}", end="")
            sys.stdout.flush()
            already_think = True
            out_tokens = []

            occurrence = {}
            for i in range(500):
                for n in occurrence:
                    think_out[n] -= (
                        self.dynamic_settings.GEN_alpha_presence
                        + occurrence[n] * self.dynamic_settings.GEN_alpha_frequency
                    )
                token = self.pipeline.sample_logits(
                    think_out,
                    temperature=self.dynamic_settings.GEN_TEMP,
                    top_p=self.dynamic_settings.GEN_TOP_P,
                )
                if token not in occurrence:
                    occurrence[token] = 1
                else:
                    occurrence[token] += 1

                think_out, think_state = self.RWKV_infer([token], think_state)
                out_tokens += [token]
                if token in self.detect_eos_token:
                    _, think_state = self.RWKV_infer(self.stop_add_token, think_state)
                    out_tokens.pop()
                    break

                think_out[self.END_OF_TEXT] = -999999999  # disable <|endoftext|>

                xxx = self.pipeline.decode([token])

                print(xxx, end="")
                sys.stdout.flush()

            print("\n↑think\n")
            send_msg = self.pipeline.decode(out_tokens)

            think_conversation = Conversation(
                f"({to_char}",
                send_msg,
                sos_tokens=self.think_sos_tokens,
                eos_tokens=self.think_eos_tokens,
                sos="",
                eos="",
            )
            self.conversation_hist.append(think_conversation)
            self.now_state = think_state
        return already_think

    def estimate_desire(self, tokens, skip_line=True, chat_token_decay=0):
        bc_len = len(tokens)

        out, state = self.RWKV_infer(
            conversation.to_tokens(self.encode),
            self.now_state,
        )
        out[self.END_OF_TEXT] = -999999999  # disable <|endoftext|>
        now_token = self.pipeline.sample_logits(
            out,
            temperature=self.dynamic_settings.GEN_TEMP,
            top_p=self.dynamic_settings.GEN_TOP_P,
        )
        if skip_line:
            while now_token in [11, 261, 263]:
                out, state = self.RWKV_infer(
                    [now_token],
                    state,
                )
                out[self.END_OF_TEXT] = -999999999  # disable <|endoftext|>

                now_token = self.pipeline.sample_logits(
                    out,
                    temperature=self.dynamic_settings.GEN_TEMP,
                    top_p=self.dynamic_settings.GEN_TOP_P,
                )
        for i in range(bc_len):
            out[chat_sos_tokens[1]] -= chat_token_decay
            now_token = self.pipeline.sample_logits(
                out,
                temperature=self.dynamic_settings.GEN_TEMP,
                top_p=self.dynamic_settings.GEN_TOP_P,
            )
            # print("-> estimate:",now_token)
            if not (now_token == tokens[i] or (now_token == 19133 and tokens[i] == 41)):
                break

            out, state = self.RWKV_infer(
                [now_token],
                state,
            )
            if i == bc_len - 1:
                return True, state, out
        return False, None, out


# test script
if __name__ == "__main__":
    model_path = "/home/li/MachineLr/RWKV-My-API/models/RURIv2929.pth"
    model_path = "/home/li/MachineLr/RWKV-My-API/models/930RURIv2.pth"
    # model_path = "/home/li/MachineLr/RWKV-LM/RWKV-v4neo/checkpoints/rwkv-10.pth"

    scroll_tokens = 0
    temp = 1
    top_p = 0.7
    GEN_alpha_frequency = 0.2
    GEN_alpha_presence = 0.2
    think_desire = 4.5
    character_name = "琉璃"
    strategy = "cuda:1 bf16"  # RWKV Strategy
    # vocab="rwkv_vocab_v20230424"
    vocab = "/home/li/MachineLr/RWKV-LM/rwkv_vocab_ssg_eval.txt"
    # character_name='测试'
    chatbot = Chatbot(
        model_path=model_path,
        scroll_tokens=scroll_tokens,
        GEN_TEMP=temp,
        GEN_TOP_P=top_p,
        character_name=character_name,
        GEN_alpha_presence=GEN_alpha_presence,
        GEN_alpha_frequency=GEN_alpha_frequency,
        sos_tokens=chat_sos_tokens,
        eos_tokens=chat_eos_tokens,
        stop_add_token=stop_add_token,
        detect_eos_token=detect_eos_token,
        vocab=vocab,
        strategy=strategy,
        think_desire=think_desire
    )
    to = chatbot.bot
    while True:
        msg = prompt("\n>")
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
            conversation = Conversation(
                character=char,
                text=real_msg,
                sos_tokens=chat_sos_tokens,
                eos_tokens=chat_eos_tokens,
            )
            chatbot.chat(conversation, to_char=to)
        elif msg.lower() == "/listen":
            chatbot.chat(Conversation(only_text=True, text=""), to_char=to)
        elif msg[:8].lower() == "/listen ":
            to = msg[8:].strip()
            chatbot.chat(Conversation(only_text=True, text=""), to_char=to)
        elif msg.lower() == "/think":
            if to is None:
                to = chatbot.bot
            chatbot.chat(
                Conversation(only_text=True, text=""),
                to_char=f"({to}",
                expect_sp_token=think_sos_tokens,
            )
        elif msg[:7].lower() == "/think ":
            to = "(" + msg[7:].strip()
            chatbot.chat(
                Conversation(only_text=True, text=""),
                to_char=f"{to}",
                expect_sp_token=think_sos_tokens,
            )
        elif msg[:5].lower() == "/sys ":
            real_msg = msg[5:].strip()
            conversation = Conversation(
                character=chatbot.system,
                text=real_msg,
                sos_tokens=system_sos_tokens,
                eos_tokens=system_eos_tokens,
            )
            chatbot.chat(conversation)
        elif msg[:5].lower() == "/add ":
            char = msg[5:].split(" ")[0]
            real_msg_s = msg[5:].split(" ")[1:]
            real_msg = ""
            for s in real_msg_s:
                real_msg += s + " "
            real_msg = real_msg.strip()
            conversation = Conversation(
                character=char,
                text=real_msg,
                sos_tokens=chat_sos_tokens,
                eos_tokens=chat_eos_tokens,
            )
            chatbot.add_fake_conversation(conversation)
        elif msg[:7].lower() == "/setup ":
            msg = msg[7:]
            chatbot.setup_prompt(msg)
        elif msg[:3].lower() == "/g ":
            msg = msg[3:]
            t = chatbot.interface
            chatbot.interface = ""
            chatbot.chat(Conversation(only_text=True, text=msg), to_char="")
            chatbot.interface = t
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
        elif msg[:9].lower() == "/svstate ":
            name = msg[9:].strip()
            chatbot.save_state(chatbot.now_state, name)
        elif msg[:9].lower() == "/ldstate ":
            name = msg[9:].strip()
            chatbot.load_state(chatbot.now_state, name)
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
            print(chatbot.calc_ctx())
        elif msg.lower() == "/svhist":
            current_time = time.localtime()
            time_string = time.strftime(r"%m-%d %H:%M:%S", current_time)
            name = time_string + ".txt"
            path = "savehist"
            if not os.path.exists(path):
                os.mkdir(path)
            with open(os.path.join(path, name), "w", encoding="utf-8") as f:
                f.write(
                    chatbot.conversation2text(
                        chatbot.change_converlist_eos(
                            chatbot.conversation_hist,
                            new_sos="<sos>",
                            new_eos="<eos>\n\n",
                        )
                    )
                )
        elif msg.lower() == "/args":
            print(chatbot.dynamic_settings.to_string())
        else:
            if "--to " in msg:
                to = msg.split("--to ")[1]
                msg = msg.split("--to ")[0]
            else:
                to = None
            conversation = Conversation(
                character=chatbot.user,
                text=msg,
                sos_tokens=chat_sos_tokens,
                eos_tokens=chat_eos_tokens,
            )
            chatbot.chat(conversation, to_char=to)
