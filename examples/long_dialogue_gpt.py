import os
import sys
print(os.getcwd())
sys.path.insert(0, os.getcwd())
# print(sys.path)
from qwen_agent.agents import DialogueRetrievalAgent
from qwen_agent.gui import WebUI


def test():
    # Define the agent
    bot = DialogueRetrievalAgent(llm={'model': 'phi-3.5-mini',
                                      'model_type':'oai',
                                      'base_url': 'https://8042-178-57-73-146.ngrok-free.app',
                                      })

    # Chat
    long_text = """Танец Bop – это уличный танцевальный стиль, который возник в середине 2010-х годов в Чикаго, США. Он является частью культуры хип-хопа и получил широкую популярность благодаря местной музыке, называемой bop music, представляющей собой легкий и ритмичный поджанр чикагского дрилла.

Основные особенности танца Bop:
Движения ног:

Основное внимание уделяется быстрым и плавным движениям ног. Танец включает прыжки, шаги и подпрыгивания, которые создают ощущение "пружинистого" ритма.
Характерные движения, такие как "D Low Shuffle", популяризированы танцорами.
Свободный стиль:

Как и другие уличные танцы, Bop поощряет импровизацию, что позволяет танцорам выражать свою индивидуальность.
Движения обычно следуют за ритмом и настроением музыки.
Энергичность и позитив:

Танец носит ярко выраженный позитивный и энергичный характер, что делает его привлекательным для молодежи.
Популяризация:

Танец стал известным благодаря местным артистам, таким как Lil Kemo и Dlow, которые активно продвигали стиль через социальные сети и видеоклипы. Одним из самых ярких примеров является вирусный танец на трек "Dlow Shuffle"."""

    messages = [{'role': 'user', 'content': f'О чём этот текст？\n{long_text}'}]

    response = bot.run(messages)
    for r in response:
        print('bot response:', r)


def app_tui():
    bot = DialogueRetrievalAgent(llm={'model': 'qwen-max'})

    # Chat
    messages = []
    while True:
        query = input('user question: ')
        messages.append({'role': 'user', 'content': query})
        response = []
        for response in bot.run(messages=messages):
            print('bot response:', response)
        messages.extend(response)


def app_gui():
    # Define the agent
    bot = DialogueRetrievalAgent(llm={'model': 'qwen-max'})

    WebUI(bot).run()


if __name__ == '__main__':
    test()
    # # app_tui()
    # app_gui()
