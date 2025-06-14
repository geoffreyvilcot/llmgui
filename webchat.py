import gradio as gr

import pickle
import numpy as np
import json
import time

from threading import Thread

import getopt

import sys
import urllib
from io import BytesIO
import requests
from config import Config

def query(message, history, systemprompt, user_header, assistant_header, user_name : str, bot_name : str, max_tokens, seed, temp, top_p):

    # history_str = ""
    # for h in history :
    #     sub = f"{user_header}{h[0]} {assistant_header} {h[1]}"
    #     history_str += f"{sub}"

    messages = []
    messages.append({"role": "system", "content": systemprompt})
    for m in history :
        messages.append({"role": "user", "content": m[0]} )
        messages.append({"role": "assistant", "content": m[1]})
    messages.append({"role": "user", "content": message})

    headers = {"Content-Type": "application/json"}
    in_data = {"messages": messages}
    api_url = f"{conf.external_llama_cpp_url}/apply-template"
    response = requests.post(api_url, data=json.dumps(in_data), headers=headers)

    prompt = json.loads(response.content)["prompt"]
#    prompt = f"{systemprompt}\n{history_str}\n{user_header}{message} {assistant_header}"

    

    print(prompt)

    start_t = time.time()
    api_url = f"{conf.external_llama_cpp_url}/completion"
    in_data = {"prompt": prompt, "n_predict": max_tokens, "seed": seed, "stream" : True, "temperature" : temp, "top_p" : top_p}


    print(f"sending : {in_data}")
    response = requests.post(api_url, data=json.dumps(in_data), headers=headers, stream=True)
    response_text = ""
    # response_text = "--\n"
    idx = 0
    start_t = None
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8').replace("data: ", "")
            j_str = json.loads(decoded_line)
            if j_str['stop'] == "true":
                print("-- STOP --")
                return
            response_text += j_str['content'].replace('>', '\\>').replace('<', '\\<')
            print(response_text)
            idx += 1
            yield response_text



if __name__ == "__main__":

    # demo = gr.Interface(
    #     allow_flagging='never',
    #     fn=query,
    #     analytics_enabled=False,
    #     inputs=[gr.Textbox(label="Url", value="http://127.0.0.1:8080"),
    #             gr.Textbox(label="Pre Prompt", lines=5),
    #             gr.Textbox(label="Inputs", lines=10),
    #             gr.Textbox(label="Stop word"),
    #             gr.Number(512, label="Max tokens"),
    #             gr.Number(-1, label="Seed"),
    #             ],
    #     outputs=[gr.Textbox(label="Outputs", lines=30), gr.Label(label="Stats")],
    #
    # )

    # demo.launch(server_name="127.0.0.1", server_port=49288)
    # auth=("admin", "pass1234")

    conf_file_name = "config.json"

    opts, args = getopt.getopt(sys.argv[1:],"hc:")
    for opt, arg in opts:
        if opt == '-h':
            print(sys.argv[0] + ' -c <conf_file>')
            sys.exit()
        elif opt in ("-c"):
            conf_file_name = arg

    conf = Config(conf_file=conf_file_name)
    with open(conf.system_prompt, "r", encoding='utf-8') as f:
        system_prompt = f.read()
    with open(conf.user_header, "r", encoding='utf-8') as f:
        user_header = f.read()
    with open(conf.assistant_header, "r", encoding='utf-8') as f:
        assistant_header = f.read()

#     pre_prompt = """This is a transcript of a dialog between User and Llama.
# Llama is a friendly chatbot with a huge knowledge.
# Llama is honest, answer with exactitude and precision."""

    gr.ChatInterface(query,
                     analytics_enabled=False,
                     additional_inputs=[
                         # gr.Textbox(label="Url", value=conf.external_llama_cpp_url),
                         gr.Textbox(label="System Prompt", lines=5, value=system_prompt),
                         gr.Textbox(label="user_header", lines=5, value=user_header),
                         gr.Textbox(label="assistant_header", lines=5, value=assistant_header),
                         gr.Textbox(label="User name", value=conf.user_name),
                         gr.Textbox(label="Bot name", value=conf.bot_name),
                         gr.Number(20480, label="Max tokens"),
                         gr.Number(-1, label="Seed"),
                         gr.Number(0.7, label="temp"),
                         gr.Number(0.95, label="top_p"),
                     ]
                     ).launch(server_name=conf.listen_bind, server_port=conf.listen_port, root_path=conf.root_path)