from alicebot import Plugin
import random
import numpy as np
import auto_response as R
import chain_dict
from qbot_event_chain import Conversation

from rwkv_interface import chatbot,temp_conversations


class AutoSpeak(Plugin):
    __schedule__ = True
    trigger = "interval"
    trigger_args = {"seconds": 180}


    async def handle(self) -> None:
        if not R.AutoResponse.already_speak:
            print("自动对话 测试")
            # 这一行需要加一个读取存档，如果没有存档则回归init_state
            if len(temp_conversations) != 0:
                chatbot.setup_prompt(
                    chatbot.conversation2text(temp_conversations)
                )
            response = chain_dict.思考是否发言.chain(chatbot,first_node=True)
            if response is not None:
                await self.bot.get_adapter("cqhttp").send(
                    response.strip(), message_type="group", id_=R.AutoResponse.listen_group
                )
                response_conversation = Conversation(
                    character=chatbot.bot,
                    text=response.strip(),
                    interface=chatbot.interface,
                    sos=chatbot.sos,
                    eos=chatbot.eos,
                )
                print(f"回复:{response}")
                R.AutoResponse.already_speak = True
                chain_dict.postprocess(response_conversation)

    async def rule(self) -> bool:
        if (
            self.event.adapter.name == "apscheduler"
            and type(self) == self.event.plugin_class
        ):
            next_seconds = random.randint(45, R.AutoResponse.time_upbound) + int(random.gauss(0, 50))
            print("next respnse seconds:", next_seconds)
            next_seconds = np.clip(next_seconds, 5, None)
            self.trigger_args["seconds"] = int(next_seconds)
            return True
        return False


