from qbot_event_chain import ChainNode
from chatbot import Conversation
import functools
from rwkv_interface import chatbot, temp_conversations
import gc

mismatch = ChainNode(Conversation(character="system", text="输入非法，和主人说下情况吧:"))


def postprocess(response_conversation: Conversation):
    chatbot.process_reaction_end(
        chatbot.conversation2text(temp_conversations) + response_conversation()
    )
    temp_conversations.clear()
    gc.collect()
    return None


def chain_splitter(
    callback_res,
    d: dict[str, ChainNode],
    mode="contains",
    allow_mismatch=False,
    mismatch_return=None,
):
    assert mode in ["contains", "match"]
    for k in d.keys():
        if k in callback_res:
            if mode == "contains":
                print(f"chain: contains->{k}")
                if d[k] is not None:
                    if isinstance(d[k], ChainNode):
                        return d[k].chain(chatbot)
                    else:
                        return d[k]()
                else:
                    return None
            elif callback_res[: len(k)] == k:
                print(f"chain: match->{k}")
                if d[k] is not None:
                    if isinstance(d[k], ChainNode):
                        return d[k].chain(chatbot)
                    else:
                        return d[k]()
                else:
                    return None
    print("chain mismatched")
    if allow_mismatch:
        return mismatch_return()
    else:
        return mismatch.chain(chatbot)


# 发言 = ChainNode(
#     conversation=Conversation(character="system", text="请输出发言内容:"),
#     preaction=functools.partial(
#         postprocess, response_conversation=Conversation(only_text=True, text="")
#     ),
# )
发言 = ChainNode(
    conversation=Conversation(only_text=True, text=""),
    preaction=functools.partial(
        postprocess,
        response_conversation=Conversation(only_text=True, text=""),
    ),
)

确认是否发言 = ChainNode(
    Conversation(character="system", text='请输出"是"或者"否"确认是否需要发言:'),
    action=functools.partial(
        chain_splitter,
        d={
            "是": 发言,
            "否": functools.partial(
                postprocess,
                response_conversation=Conversation(only_text=True, text=""),
            ),
        },
        mode="contains",
        allow_mismatch=True,
        mismatch_return=functools.partial(
            postprocess,
            response_conversation=Conversation(only_text=True, text=""),
        ),
    ),
)

学习完毕 = ChainNode(
    Conversation(
        character="system",
        text='已加入训练队列，请输出"是"或者"否"确认是否还需要发言:',
        sos=chatbot.sos,
        eos=chatbot.eos,
    ),
    action=functools.partial(
        chain_splitter, d={"是": 发言, "否": None}, mode="contains", allow_mismatch=True
    ),
)
学习 = ChainNode(
    Conversation(
        character="system",
        text="请总结需要学习的内容并输出:",
        sos=chatbot.sos,
        eos=chatbot.eos
    ),
    next_node=学习完毕,
)

确认是否发言与学习 = ChainNode(
    Conversation(character="system", text='请输出"是"或者"否"确认是否需要发言，也可以输出"学习"来学习聊天中提到的新知识:'),
    action=functools.partial(
        chain_splitter,
        d={
            "是": 发言,
            "否": functools.partial(
                postprocess,
                response_conversation=Conversation(only_text=True, text=""),
            ),
            "学习": 学习,
        },
        mode="contains",
        allow_mismatch=True,
        mismatch_return=functools.partial(
            postprocess,
            response_conversation=Conversation(only_text=True, text=""),
        ),
    ),
)

# 思考是否发言 = ChainNode(
#     Conversation(character='system', text="思考一下现在是否需要发言，以及聊天中是否有有价值的知识。输出思考过程："), next_node=确认是否发言
# )
思考是否发言 = ChainNode(
    Conversation(
        character="system",
        text="思考一下现在是否需要发言，并输出思考过程：",
        sos=chatbot.sos,
        eos=chatbot.eos,
    ),
    next_node=确认是否发言,
)
