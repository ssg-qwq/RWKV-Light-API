from chatbot import Chatbot


model_path = "../models/merge7-29-noeos-interesting"
# model_path = "../models/merge8-5-neweos-nice"
prompt_path= "../myprompt.py"
character_name = '琉璃_群聊'
scroll_tokens = 0
temp = 1
top_p = 0.85
sos=''
eos='\n\n'
detect_eos='\n\n'
vocab="rwkv_vocab_v20230424"

# sos = "<|st|>"
# eos = "<|ed|>\n"
# detect_eos = "<|ed|>"
# vocab="rwkv_vocab_v20230424addeos"

chatbot = Chatbot(
    model_path=model_path,
    scroll_tokens=scroll_tokens,
    GEN_TEMP=temp,
    GEN_TOP_P=top_p,
    character_name=character_name,
    prompt_path=prompt_path,
    sos=sos,
    eos=eos,
    detect_eos=detect_eos,
    vocab=vocab
)

temp_conversations=[]
