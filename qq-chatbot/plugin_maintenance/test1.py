from alicebot import Plugin


class Test1(Plugin):
    async def handle(self) -> None:
        if (
            self.bot.config.nickname in self.event.message.get_plain_text()
            or str(self.bot.config.bot_id) in self.event.message
        ):
            await self.event.reply('啊咧咧，琉璃好像在维护中……')

    async def rule(self) -> bool:
        return self.event.message_type == "group"
