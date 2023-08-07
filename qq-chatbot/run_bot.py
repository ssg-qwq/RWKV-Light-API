from alicebot import Bot
import sys


sys.path.append('./plugins_auto')
sys.path.append('./plugins')
sys.path.append('..')

if __name__ == "__main__":
    bot = Bot()
    bot.run()