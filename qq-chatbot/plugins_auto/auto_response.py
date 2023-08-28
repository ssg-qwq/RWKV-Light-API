from alicebot import Plugin
import chain_dict
from chatbot import Conversation
from rwkv_interface import chatbot, temp_conversations
import time
import random
import re
import os


def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


class AutoResponse(Plugin):
    already_speak = False
    listen_group = 368175492
    rand_speak_prob_max = 10

    knowledge_hist_len = 10
    avoid_bot_ids = ["1875153583"]
    avoid_times = 3  # 屏蔽bot出现超过n次则不记忆
    knowledge_prob = 3

    time_upbound = 225

    data_collection_turn = 50  # 单次存储轮数
    data_collection_groups = [439087067, 762489654, 584182623]  # 收集数据的群聊
    data_collection_groups_count_dict = {}
    data_collection_groups_conversation_dict = {}

    async def handle(self) -> None:
        print("group:", self.event.group_id, type(self.event.group_id))
        # print("listen:", self.listen_group)
        print("subtype", self.event.sub_type)
        msg = self.event.message.get_plain_text()
        # 重置
        if self.event.user_id == self.bot.config.superuser and "*reset" in msg:
            chatbot.reset()
            temp_conversations.clear()
            await self.event.reply("已重置。")
        # 更改监听群聊
        elif self.event.user_id == self.bot.config.superuser and "*listengrp " in msg:
            if is_number(msg.split("*listengrp ")[1]):
                AutoResponse.listen_group = int(msg.split("*listengrp ")[1])
        # 更改发言概率
        elif self.event.user_id == self.bot.config.superuser and "*prob " in msg:
            if is_number(msg.split("*prob ")[1]):
                AutoResponse.rand_speak_prob_max = int(msg.split("*prob ")[1])
        # 更改发言概率
        elif self.event.user_id == self.bot.config.superuser and "*upbound " in msg:
            if is_number(msg.split("*upbound ")[1]):
                AutoResponse.time_upbound = int(msg.split("*upbound ")[1])
        # 系统消息
        elif self.event.user_id == self.bot.config.superuser and "*sys " in msg:
            c = Conversation(
                character=chatbot.system,
                text=msg.split("*sys ")[1],
                sos=chatbot.sos,
                eos=chatbot.eos,
            )
            response = chatbot.chat(c)
            await self.event.reply(response.strip())
        # 后台输出历史
        elif self.event.user_id == self.bot.config.superuser and "*hist" in msg:
            for c in chatbot.conversation_hist:
                print(
                    "conversation_hist:",
                    c().replace("\r\n", "\n").replace(chatbot.detect_eos, "\n\n"),
                )
            for c in temp_conversations:
                print(
                    "hist:",
                    c().replace("\r\n", "\n").replace(chatbot.detect_eos, "\n\n"),
                )
        # 存档
        elif self.event.user_id == self.bot.config.superuser and "*save " in msg:
            if msg.split("*save ")[1] != "":
                chatbot.save_history(msg.split("*save ")[1])
        # 读档
        elif self.event.user_id == self.bot.config.superuser and "*load " in msg:
            if msg.split("*load ")[1] != "":
                try:
                    chatbot.load_history(msg.split("*load ")[1])
                    temp_conversations.clear()
                except:
                    print("读取失败")
        # 设置参数
        elif self.event.user_id == self.bot.config.superuser and "*set" in msg:
            if "-temp=" in msg:
                arg = float(msg.split("-temp=")[1].split(" ")[0])
                chatbot.dynamic_settings.GEN_TEMP = arg
                await self.event.reply(f"temp={arg}")
            if "-top_p=" in msg:
                arg = float(msg.split("-top_p=")[1].split(" ")[0])
                chatbot.dynamic_settings.GEN_TOP_P = arg
                await self.event.reply(f"top_p={arg}")
            if "-alpha_presence=" in msg:
                arg = float(msg.split("-alpha_presence=")[1].split(" ")[0])
                chatbot.dynamic_settings.GEN_alpha_presence = arg
                await self.event.reply(f"alpha_presence={arg}")
            if "-alpha_frequency=" in msg:
                arg = float(msg.split("-alpha_frequency=")[1].split(" ")[0])
                chatbot.dynamic_settings.GEN_alpha_frequency = arg
                await self.event.reply(f"alpha_frequency={arg}")
        # 在群聊内说话逻辑
        elif int(self.event.group_id) == self.listen_group:
            rand = random.randint(0, 100) < random.randint(
                1, AutoResponse.rand_speak_prob_max
            )
            if "*+++" in msg:
                if len(temp_conversations) > 0:
                    chatbot.setup_prompt(chatbot.conversation2text(temp_conversations))
                response = chain_dict.思考是否发言.chain(chatbot, first_node=True)
                if response is not None:
                    await self.event.reply(response.strip())
                    response_conversation = Conversation(
                        character=chatbot.bot,
                        text=response.strip(),
                        interface=chatbot.interface,
                        sos=chatbot.sos,
                        eos=chatbot.eos,
                    )
                    AutoResponse.already_speak = True
                    chatbot.process_reaction_end(
                        chatbot.conversation2text(temp_conversations)
                        + response_conversation()
                    )
                    temp_conversations.clear()
            # 直接对话
            elif (
                int(self.event.group_id) == self.listen_group
                and self.event.message.get_plain_text()[:2] == "**"
            ):
                self.append_qqmessage2list(
                    temp_conversations, message=str(self.event.message)[2:]
                )
                # current_time = time.time()
                # local_time = time.localtime(current_time)
                # time_string = time.strftime("%H:%M:%S ", local_time)
                # if self.event.user_id == self.bot.config.superuser:
                #     temp_conversations.append(
                #         Conversation(
                #             character=f"{self.event.sender.nickname}(主人)({self.event.user_id}) {time_string}",
                #             text=self.remove_duplicate_qq_usernames(
                #                 self.cqat2str(
                #                     str(self.event.message)[2:],
                #                     match_qq=self.bot.config.bot_id,
                #                 )
                #             ),
                #             sos=chatbot.sos,
                #             eos=chatbot.eos,
                #         )
                #     )
                # else:
                #     temp_conversations.append(
                #         Conversation(
                #             character=f"{self.event.sender.nickname}({self.event.user_id}) {time_string}",
                #             text=self.remove_duplicate_qq_usernames(
                #                 self.cqat2str(
                #                     str(self.event.message)[2:],
                #                     match_qq=self.bot.config.bot_id,
                #                 )
                #             ),
                #             sos=chatbot.sos,
                #             eos=chatbot.eos,
                #         )
                #     )
                if len(temp_conversations) > 0:
                    chatbot.setup_prompt(chatbot.conversation2text(temp_conversations))
                response = chain_dict.发言.chain(chatbot, first_node=True)
                if response is not None:
                    await self.bot.get_adapter("cqhttp").send(
                        response.strip(),
                        message_type="group",
                        id_=AutoResponse.listen_group,
                    )
                    response_conversation = Conversation(
                        character=chatbot.bot,
                        text=response.strip(),
                        interface=chatbot.interface,
                        sos=chatbot.sos,
                        eos=chatbot.eos,
                    )
                    AutoResponse.already_speak = True
                    chatbot.process_reaction_end(
                        chatbot.conversation2text(temp_conversations)
                        + response_conversation()
                    )
                    temp_conversations.clear()
            elif (
                self.bot.config.nickname in self.event.message.get_plain_text()
                or str(self.bot.config.bot_id) in self.event.message
                or rand
            ):
                self.append_qqmessage2list(
                    temp_conversations, message=str(self.event.message)
                )
                # current_time = time.time()
                # local_time = time.localtime(current_time)
                # time_string = time.strftime("%H:%M:%S ", local_time)
                # if self.event.user_id == self.bot.config.superuser:
                #     temp_conversations.append(
                #         Conversation(
                #             character=f"{self.event.sender.nickname}(主人)({self.event.user_id}) {time_string}",
                #             text=self.remove_duplicate_qq_usernames(
                #                 self.cqat2str(
                #                     str(self.event.message),
                #                     match_qq=self.bot.config.bot_id,
                #                 )
                #             ),
                #             sos=chatbot.sos,
                #             eos=chatbot.eos,
                #         )
                #     )
                # else:
                #     temp_conversations.append(
                #         Conversation(
                #             character=f"{self.event.sender.nickname}({self.event.user_id}) {time_string}",
                #             text=self.remove_duplicate_qq_usernames(
                #                 self.cqat2str(
                #                     str(self.event.message),
                #                     match_qq=self.bot.config.bot_id,
                #                 )
                #             ),
                #             sos=chatbot.sos,
                #             eos=chatbot.eos,
                #         )
                #     )
                if len(temp_conversations) > 0:
                    chatbot.setup_prompt(chatbot.conversation2text(temp_conversations))
                response = chain_dict.思考是否发言.chain(chatbot, first_node=True)
                if response is not None:
                    await self.event.reply(response.strip())
                    response_conversation = Conversation(
                        character=chatbot.bot,
                        text=response.strip(),
                        interface=chatbot.interface,
                        sos=chatbot.sos,
                        eos=chatbot.eos,
                    )
                    AutoResponse.already_speak = True
                    chatbot.process_reaction_end(
                        chatbot.conversation2text(temp_conversations)
                        + response_conversation()
                    )
                    temp_conversations.clear()
                # 记录历史
                if rand or random.randint(0, 100) <= AutoResponse.knowledge_prob:
                    self.add_hist2dataset()
            else:
                self.append_qqmessage2list(
                    temp_conversations, message=str(self.event.message)
                )
                # current_time = time.time()

                # local_time = time.localtime(current_time)
                # time_string = time.strftime("%H:%M:%S ", local_time)
                # if self.event.user_id == self.bot.config.superuser:
                #     temp_conversations.append(
                #         Conversation(
                #             character=f"{self.event.sender.nickname}(主人)({self.event.user_id}) {time_string}",
                #             text=self.event.message.get_plain_text(),
                #             sos=chatbot.sos,
                #             eos=chatbot.eos,
                #         )
                #     )
                # else:
                #     temp_conversations.append(
                #         Conversation(
                #             character=f"{self.event.sender.nickname}({self.event.user_id}) {time_string}",
                #             text=self.event.message.get_plain_text(),
                #             sos=chatbot.sos,
                #             eos=chatbot.eos,
                #         )
                #     )
                for line in chatbot.conversation_hist:
                    print(line())
                print(self.event.message)
                AutoResponse.already_speak = False

        # 数据收集
        elif int(self.event.group_id) in AutoResponse.data_collection_groups:
            f_name = f"./data_collection/{self.event.group_id}.txt"
            key = str(self.event.group_id)
            if key not in AutoResponse.data_collection_groups_count_dict:
                AutoResponse.data_collection_groups_count_dict[key] = 1
            else:
                AutoResponse.data_collection_groups_count_dict[key] += 1
            with open(f_name, "a", encoding="utf-8") as f:
                if AutoResponse.data_collection_groups_count_dict[key] == 1:
                    f.write("====群聊记录====\n\n")
                    AutoResponse.data_collection_groups_conversation_dict[key] = []
                self.append_qqmessage2list(
                    AutoResponse.data_collection_groups_conversation_dict[key],
                    message=str(self.event.message),
                )
                if (
                    AutoResponse.data_collection_groups_count_dict[key]
                    == AutoResponse.data_collection_turn
                ):
                    AutoResponse.data_collection_groups_count_dict[key] = 0
                    f.write(
                        chatbot.conversation2text(
                            chatbot.change_converlist_eos(
                                AutoResponse.data_collection_groups_conversation_dict[
                                    key
                                ],
                                new_eos="<eos>\n\n",
                                new_sos="<sos>",
                            )
                        )
                    )
                    f.write("====记录结束====\n\n")

    async def rule(self) -> bool:
        if self.event.adapter.name != "mirai" and self.event.adapter.name != "cqhttp":
            return False
        if self.event.type != "message":
            return False
        return self.event.message_type == "group"

    def cqat2str(self, text, match_qq):
        # 定义正则表达式模式
        pattern = r"\[(.*?)\]|CQ:at,qq=(\d+)"

        # 定义匹配函数，用于处理匹配结果
        def replace_function(match):
            if match.group(1):
                # 如果匹配到了"[]"内的内容，进行处理
                inner_content = match.group(1)
                if "CQ:at,qq=" in inner_content:
                    # 如果"[]"内部包含CQ:at，则保留该内容，并添加@符号
                    qq_number = inner_content.split("CQ:at,qq=")[1]
                    if qq_number == str(match_qq):
                        return f"@{qq_number}({self.bot.config.nickname})"
                    else:
                        return f"@{qq_number}"
                elif "CQ:image,file=" in inner_content:
                    return "(图片表情)"
                elif "CQ:record,file=" in inner_content:
                    return "(语音消息)"
                else:
                    # 否则，删除"[]"内部的内容
                    return ""

        # 使用sub函数进行正则替换
        result = re.sub(pattern, replace_function, text)

        return result

    def remove_duplicate_qq_usernames(self, text):
        pattern = r"(@(\d+)\s*\([^)]+\)) (\1)+"
        result = re.sub(pattern, r"\1", text)
        return result

    def str2cqat(self, input_str):
        # 定义正则表达式模式，匹配[@qq号]格式
        pattern = r"\[@(\d+)\]"

        # 定义匹配函数，用于处理匹配结果
        def replace_function(match):
            qq_number = match.group(1)
            return f"[CQ:at,qq={qq_number}]"

        # 使用sub函数进行正则替换
        result = re.sub(pattern, replace_function, input_str)

        return result

    def add_hist2dataset(self, dataset_path="./memory.txt"):
        print("Add Memory...")
        with open(dataset_path, "a", encoding="utf-8") as file:
            cs = chatbot.conversation_hist + temp_conversations
            memory = (
                cs[-AutoResponse.knowledge_hist_len :]
                if len(cs) > AutoResponse.knowledge_hist_len
                else cs
            )
            count = 0
            for c in memory:
                for id in AutoResponse.avoid_bot_ids:
                    if id in c.character or id in c.text:
                        count += 1
            if count < AutoResponse.avoid_times:
                memory = f"========================\n{chatbot.conversation2text(memory)}========================\n\n\n"
                file.write(memory)

    def reduce_newlines(self, text):
        reduced_text = re.sub(r"\n{3,}", "\n\n", text)
        return reduced_text

    def append_qqmessage2list(self, conversation_list, message, superuser_n="主人"):
        current_time = time.time()
        local_time = time.localtime(current_time)
        time_string = time.strftime("%H:%M:%S ", local_time)
        if self.event.user_id == self.bot.config.superuser:
            conversation_list.append(
                Conversation(
                    character=f"{self.event.sender.nickname}({superuser_n})({self.event.user_id}) {time_string}",
                    text=self.remove_duplicate_qq_usernames(
                        self.cqat2str(message, match_qq=self.bot.config.bot_id)
                    ),
                    sos=chatbot.sos,
                    eos=chatbot.eos,
                )
            )
        else:
            conversation_list.append(
                Conversation(
                    character=f"{self.event.sender.nickname}({self.event.user_id}) {time_string}",
                    text=self.remove_duplicate_qq_usernames(
                        self.cqat2str(message, match_qq=self.bot.config.bot_id)
                    ),
                    sos=chatbot.sos,
                    eos=chatbot.eos,
                )
            )
