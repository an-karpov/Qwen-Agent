import os
import sys
import pathlib
PROJ_DIR = pathlib.Path("__file__").parent.parent.parent.absolute()
BENCH_DIR = PROJ_DIR.parent/"LongContext"
sys.path.insert(0, PROJ_DIR.as_posix())
sys.path.insert(1, BENCH_DIR.as_posix())
sys.path.insert(1, (BENCH_DIR/"dataloaders").as_posix())

from dataloaders.globalset import GlobalSet

from qwen_agent.agents import DialogueRetrievalAgent
from qwen_agent.gui import WebUI
from benchmark.longbench.model import Phi3Model

def qa_inf_bench(task, llm):
    """process a task QA bases long context"""

    bot = DialogueRetrievalAgent(llm=llm)

    if type != "qa":
        raise ValueError("incorrect task type")

    messages = [
        {'role': 'system', 'content': str(task.content)},
        {'role': 'user', 'content': str(task.question)}
        ]

    for response in bot.run(messages):
        print('bot response:', response)
    print("Correct answer: ", task.answer)

def test_qwen_qa():
    pass

def test_phi3_qa():
    llm = Phi3Model()

# if __name__ == "__main__":
#     print("ok")
