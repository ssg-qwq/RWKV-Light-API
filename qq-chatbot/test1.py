from alicebot import Plugin


class Test1(Plugin):
    async def handle(self) -> None:
        if (
            self.bot.config.nickname in self.event.message.get_plain_text()
            or str(self.bot.config.bot_id) in self.event.message
        ):
            attributes = dir(self.event)
            for attribute in attributes:
                print(attribute)
            print("检测到输入", self.event.message)
            print("检测到输入", str(self.event.message))
            print("user_id", self.event.user_id)
            print("message_type", self.event.message_type)
            print("get", self.event.get)
            print("dict", self.event.dict)
            print("at",'at' in self.event.message)
            # await self.event.reply('[CQ:at,qq=870326915]')

    async def rule(self) -> bool:
        return self.event.message_type == "group"
