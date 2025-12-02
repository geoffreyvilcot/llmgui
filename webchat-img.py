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
import os, base64
from PIL import Image

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

def query(message, history, img_path, systemprompt, max_tokens, seed, temp, top_p):
    gradio_response = ""
    messages = []
    messages.append({"role": "system", "content": systemprompt})
    messages = messages + history

    img_b64 = None
    if img_path:
        try:
            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as img:
                        if img.mode not in ("RGB", "RGBA"):
                            img = img.convert("RGB")
                        max_size = 512
                        w, h = img.size
                        scale = min(max_size / w, max_size / h, 1.0)
                        new_size = (int(w * scale), int(h * scale))
                        if new_size != img.size:
                            img = img.resize(new_size, Image.LANCZOS)
                        buf = BytesIO()
                        img.save(buf, format="PNG")
                        img_bytes = buf.getvalue()
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
            else:
                print(f"Image not found: {img_path}")
        except Exception as e:
            print(f"Error encoding image {img_path}: {e}")

    user_content = [{"type": "text", "text": message}]
    if img_b64:
        user_content += [{"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]
    messages.append({"role": "user", "content": user_content})

    # headers = {"Content-Type": "application/json"}
    # in_data = {"messages": messages}
    # api_url = f"{conf.external_llama_cpp_url}/apply-template"
    # response = requests.post(api_url, data=json.dumps(in_data), headers=headers)

    # prompt = json.loads(response.content)["prompt"]

    print(messages)

    start_t = time.time()
    api_url = f"{conf.external_llama_cpp_url}/v1/chat/completions"
    in_data = {"model": "ministral", "messages": messages, "n_predict": max_tokens, "seed": seed, "stream" : True, "temperature" : temp, "top_p" : top_p}
    response = requests.post(api_url, data=json.dumps(in_data),  stream=True)
    response_text = ""
    
    idx = 0
    start_t = None
    for line in response.iter_lines():
        if line:
            print(line)
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
            print(response_text)
            idx += 1
            
            yield message, history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response_text}
                ], \
                response_text
    
    yield "", history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response_text}
                ], \
        response_text


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


    shortcut_js = """
        <script>
        function shortcuts(e) {

            if (e.code == "Enter" && e.shiftKey) {
                document.getElementById("my_btn").click();
            }
        }
        document.addEventListener('keyup', shortcuts, false);
        </script>
    """
    # with gr.Blocks(head=shortcut_js) as query_interface:
    with gr.Blocks() as query_interface:
        chatbot = gr.Chatbot(resizable=True)
        with gr.Row():
            input_text = gr.Textbox(label="Input", lines=2, value="Describe the image",  placeholder="Type your message here...", scale=10)
            img_input = gr.Image(type="filepath", label="Input Image (optional)", scale=2)
            btn_process_text = gr.Button(value=">", elem_id="my_btn", scale=1)
        with gr.Accordion("Advance parameters", open=False) :
            system_prompt = gr.Textbox(label="System Prompt", lines=5, value=system_prompt)
            max_tokens = gr.Number(20480, label="Max tokens")
            seed = gr.Number(-1, label="Seed")
            temp = gr.Number(0.7, label="temp")
            top_p = gr.Number(0.95, label="top_p")
            debug_txt =gr.Textbox(label="Debug", lines=15)        
        btn_process_text.click(query, inputs=[input_text,chatbot, img_input, system_prompt, max_tokens, seed, temp, top_p], 
                               outputs=[input_text, chatbot, debug_txt])
    query_interface.analytics_enabled = False
    query_interface.title = "Chat bot interface"
    query_interface.launch(server_name=conf.listen_bind, server_port=conf.listen_port, root_path=conf.root_path)