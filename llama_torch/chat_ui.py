import time

import gradio as gr

# TODO: 需要实现多轮对话中使用工具的效果

def more_call(func: callable):
    """在初次回答后, 让模型调用工具回答"""
    def wrapper(message, history):
        resp = func(message, history)
        time.sleep(1)
        resp = tool_man(resp, history)
        return resp

    return wrapper


@more_call
def yes_man(message, history):
    """对应用户的首次提问"""
    if message.endswith("?"):
        return "Yes"
    else:
        return "Ask me anything!"


def tool_man(message, history):
    """对应工具函数的输出"""
    if message.endswith("?"):
        return message + "\n工具Yes"
    else:
        return message + "\n工具Ask me anything!"


demo = gr.ChatInterface(
    yes_man,
    chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(
        placeholder="说出你的疑惑", container=False, scale=7
    ),
    title="智慧助手",
    description="解答任何问题",
    theme="soft",
    examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
    cache_examples=False,
    retry_btn=None,
    undo_btn="删除上一个提问",
    clear_btn="清空",
).queue()

if __name__ == "__main__":
    demo.launch(server_port=7200)
