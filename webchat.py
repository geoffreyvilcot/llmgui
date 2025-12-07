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

import re

def supprimer_entre_balises(texte):
    # print(f"texte : {texte}")
    result = re.sub(r'\|think\|.*?\|/think\|', '', texte, flags=re.DOTALL)
    # print(f"result : {result}\n-----\n")
    return result

def supprimer_lignes_boxed(texte):
    """
    Supprime les lignes qui commencent par '\boxed{'.
    """
    result = re.sub(r'^\\boxed\{.*$', '', texte, flags=re.MULTILINE)
    # print(f"result : {result}\n-----\n")
    return result

supprimer_lignes_boxed("bla bla\n\\boxeed{coucou}\n\\boxed{coucou2}\n\\boxed{coucou3}\nfin bla bla")

def query(message, history, systemprompt, user_header, assistant_header, user_name : str, bot_name : str, max_tokens, seed, temp, top_p):

    # history_str = ""
    # for h in history :
    #     sub = f"{user_header}{h[0]} {assistant_header} {h[1]}"
    #     history_str += f"{sub}"

    messages = []
    messages.append({"role": "system", "content": systemprompt})
    for m in history :
        messages.append({"role": "user", "content": m[0]} )
        history_response_filtered = supprimer_entre_balises(m[1])
        messages.append({"role": "assistant", "content": history_response_filtered})
    messages.append({"role": "user", "content": message})

    # print(f"messages : {messages}")

    headers = {"Content-Type": "application/json"}
    in_data = {"messages": messages}
    api_url = f"{conf.external_llama_cpp_url}/apply-template"
    response = requests.post(api_url, data=json.dumps(in_data), headers=headers)

    prompt = json.loads(response.content)["prompt"]
#    prompt = f"{systemprompt}\n{history_str}\n{user_header}{message} {assistant_header}"

    # print(prompt)

    start_t = time.time()
    api_url = f"{conf.external_llama_cpp_url}/completion"
    in_data = {"prompt": prompt, "n_predict": max_tokens, "seed": seed, "stream" : True, "temperature" : temp, "top_p" : top_p}


    # print(f"sending : {in_data}")
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
            response_text += j_str['content'].replace('>', '|').replace('<', '|')
            # print(response_text)
            idx += 1
            
            yield response_text
    
    yield supprimer_lignes_boxed(response_text)

def query(message, history, systemprompt, stop_words, user_name : str, bot_name : str, max_tokens, seed, temp, top_p):
    gradio_response = ""
    system_prompt = systemprompt.replace("{{user}}", user_name).replace("{{char}}", bot_name)
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages = messages + history

    user_content = [{"type": "text", "text": message}]
    messages.append({"role": "user", "content": user_content})

    print(f"\n{messages}")

    start_t = time.time()
    api_url = f"{conf.external_llama_cpp_url}/v1/chat/completions"
    in_data = {"model": "ministral", "messages": messages, "n_predict": max_tokens, "seed": seed, "stream" : True, "temperature" : temp, "top_p" : top_p}
    response = requests.post(api_url, data=json.dumps(in_data),  stream=True)
    response_text = ""
    
    idx = 0
    start_t = None
    for line in response.iter_lines():
        if line:
            # print(line)
            decoded_line = line.decode('utf-8').replace("data: ", "")
            j_str = json.loads(decoded_line)
            # if j_str['stop'] == "true":
            #     print("-- STOP --")
            #     return
            if j_str['choices'][0]['finish_reason'] == "stop":
                print("-- STOP --")
                break
            if j_str['choices'][0]['delta']['content'] :
                response_text += j_str['choices'][0]['delta']['content']
            # print(response_text)
            idx += 1
            
            if stop_words is not None and stop_words in response_text:
                response_text = response_text.replace(stop_words, "")
                break

            # yield history + [
            #     {"role": "user", "content": message},
            #     {"role": "assistant", "content": response_text}
            #     ]
            yield response_text
    
    # yield history + [
    #             {"role": "user", "content": message},
    #             {"role": "assistant", "content": response_text}
    #             ]
    yield response_text

def call_llm_api_v1(messages, max_tokens, seed, temp, top_p):

    start_t = time.time()
    api_url = f"{conf.external_llama_cpp_url}/v1/chat/completions"
    # in_data = {"prompt": prompt, "n_predict": max_tokens, "seed": seed, "stream" : False, "temperature" : temp, "top_p" : top_p}

    in_data = {"model": "ministral", "messages": messages, "n_predict": max_tokens, "seed": seed, "stream" : False, "temperature" : temp, "top_p" : top_p}
    # print(f"sending to LLM API: {in_data}")
    response = requests.post(api_url, data=json.dumps(in_data), stream=False)

    print(response)
    
    msg_response = json.loads(response.text)['choices'][0]['message']
    print(msg_response)
    return(msg_response['content'])

if __name__ == "__main__":
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
                         gr.Textbox(label="stop_words", lines=1, value=conf.stop_words),
                         gr.Textbox(label="User name", value=conf.user_name),
                         gr.Textbox(label="Bot name", value=conf.bot_name),
                         gr.Number(20480, label="Max tokens"),
                         gr.Number(-1, label="Seed"),
                         gr.Number(0.7, label="temp"),
                         gr.Number(0.95, label="top_p"),
                     ]
                     ).launch(server_name=conf.listen_bind, server_port=conf.listen_port, root_path=conf.root_path)