import requests
import uuid
import os
import json
from langchain_gigachat import GigaChat
from langchain.schema import HumanMessage, SystemMessage
from utils import GigaChatAPIWrapper


giga = GigaChatAPIWrapper()
print('Use Model: ', giga.model)

messages = [
    SystemMessage(
        content="Ты эмпатичный бот-психолог, который помогает пользователю решить его проблемы."
    )
]

while(True):
    user_input = input("Пользователь: ")
    if user_input == "пока":
      break
    messages.append(HumanMessage(content=user_input))
    res = giga.chat.invoke(messages)
    messages.append(res)
    print("GigaChat: ", res.content)
