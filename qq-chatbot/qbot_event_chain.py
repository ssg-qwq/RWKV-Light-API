import functools

from chatbot import Chatbot, Conversation


class ChainNode:
    def __init__(
        self, conversation: Conversation, preaction=None, action=None, next_node=None
    ):
        # assert isinstance(next_node, ChainNode)
        self.conversation = Conversation
        self.next_node = next_node
        self.conversation = conversation
        self.preaction = preaction
        self.action = action

    def chain(self, bot: Chatbot, first_node=False, *args, **kwargs):
        self.conversation.sos = bot.sos
        self.conversation.eos = bot.eos
        self.conversation.interface = bot.interface

        if self.preaction is not None:
            self.preaction(*args, **kwargs)

        if self.next_node is None:
            res = bot.chat(self.conversation, mark_before_action=not first_node)
        else:
            res = bot.chat(
                self.conversation,
                mark_before_action=not first_node,
                response_callback=functools.partial(self.next_node.chain),
            )
        if self.action is not None:
            return self.action(res)
        if self.next_node is None:
            return res
        return bot.callback_res
