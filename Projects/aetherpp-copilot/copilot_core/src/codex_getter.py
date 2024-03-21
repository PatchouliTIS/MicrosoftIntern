'''
Author: majing
Description: get codex result
'''
import os
import json
import asyncio
import websockets
from utils.key_vault_util import KeyVaultHelper


class CodexGetter:
    def __init__(self, prompt, temperature=0.7, max_tokens=256, top_p=1, streaming=False, token=None):
        self.messages = []
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.streaming = streaming
        self.token = token

    async def receive_data(self, websocket):
        try:
            async for message in websocket:
                message = json.loads(message)
                if message["type"] == "status" and message["details"] == "Done":
                    # If the server sends a "Done" message, return from the coroutine
                    return
                elif message["type"] == "sse":
                    if self.streaming:
                        # print(message["payload"]["choices"][0]["text"])
                        # global messages
                        self.messages.append(message["payload"]["choices"][0]["text"])
                    else:
                        if len(self.messages) == 0:
                            self.messages.append("")
                        self.messages[0] += message["payload"]["choices"][0]["text"]
        except asyncio.CancelledError:
            # If the coroutine is cancelled, log a message and return
            print("receive_data() coroutine cancelled")
            return

    async def send_data(self, websocket):
        # Send a message to the server
        payload = {
            "prompt": self.prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
            }

        message = {
            'endpoint': "dv3",
            'token': None,
            'auth': self.token,
            'custom_headers': {},
            'payload': payload,
        }

        # Send a message to the server
        await websocket.send(json.dumps(message))

    async def connect(self):
        try:
            # Connect to the websocket server
            async with websockets.connect("wss://codexplayground.azurewebsites.net/papyrus/") as websocket:
                # Start a coroutine to handle incoming messages
                receive_task = asyncio.create_task(self.receive_data(websocket))

                # Send a message to the server
                await self.send_data(websocket)

                # Wait for all coroutines to finish
                # await receive_task
                await asyncio.gather(receive_task)
        except asyncio.CancelledError:
            # If the coroutine is cancelled, log a message and return
            print("Task cancelled")

    def run(self):
        asyncio.run(self.connect())


if __name__ == "__main__":
    # current file path
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    workspace_config_path = os.path.join(current_file_path, "..", "config", "workspace", "wxtcstrain.json")
    token = KeyVaultHelper(workspace_config_path).get_secret("codex-playground-token")

    prompt = "You are translator. You are also search engine expert. Output the first translation of the query into Chinese.\r\nQuery: How to park when visiting the Summer Palace\r\nTranslation:"

    ## test streaming
    import threading
    codex_getter = CodexGetter(prompt, temperature=0.7, max_tokens=64, top_p=1.0, streaming=True, token=token)
    t = threading.Thread(target=codex_getter.run)
    t.start()

    def consume():
        # global messages # 声明使用全局变量
        while True: # 循环读取数据
            if codex_getter.messages: # 如果数组不为空
                message = codex_getter.messages.pop(0) # 取出第一个元素，并从数组中删除
                yield message
            else: # 如果数组为空，判断子线程是否结束
                if not t.is_alive(): # 如果子线程已经结束 
                    break # 跳出循环 

    for da in consume():
        print(da, end="")

    ## test non-streaming
    codex_getter = CodexGetter(prompt, temperature=0.7, max_tokens=64, top_p=1.0, streaming=False, token=token)
    codex_getter.run()
    print("\nhello")
    print(codex_getter.messages)
