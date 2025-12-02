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

def query(url, preprompt, Inputs, stop_word : str, max_tokens, seed):

    prompt = f"{preprompt}\n{Inputs}"

    print(prompt)

    if stop_word.strip() == '' :
        json_stop = None
    else :
        json_stop = [stop_word]

    start_t = time.time()
    api_url = f"{url}/completion"
    in_data = {"prompt": prompt, "n_predict": max_tokens, "stop" : json_stop, "seed": seed, "stream" : True}

    # api_url = f"{url}/embedding"
    # in_data = {"content": prompt}

    headers = {"Content-Type": "application/json"}
    print(f"sending : {in_data}")
    global_start_t = time.time()
    response = requests.post(api_url, data=json.dumps(in_data), headers=headers, stream=True)
    response_text = ""
    idx = 0
    start_t = None
    for line in response.iter_lines():

        # filter out keep-alive new lines
        if line:
            decoded_line = line.decode('utf-8').replace("data: ", "")
            j_str = json.loads(decoded_line)
            if j_str['stop'] == "true":
                print("-- STOP --")
                return
            if start_t is None:
                start_t = time.time() - 1
                prompt_duration = time.time() - global_start_t
            response_text += j_str['content']
            end_t = time.time()
            idx += 1
            yield response_text, f"Elapsed time {end_t -start_t:0.2f}s / Prompt eval {prompt_duration:0.2f} sec / {idx} tokens / {idx/(end_t -start_t):0.2f} tokens / sec"


if __name__ == "__main__":

    demo = gr.Interface(
        fn=query,
        analytics_enabled=False,
        inputs=[gr.Textbox(label="Url", value="http://127.0.0.1:8080"),
                gr.Textbox(label="Pre Prompt", lines=5),
                gr.Textbox(label="Inputs", lines=10),
                gr.Textbox(label="Stop word"),
                gr.Number(512, label="Max tokens"),
                gr.Number(-1, label="Seed"),
                ],
        outputs=[gr.Textbox(label="Outputs", lines=30), gr.Label(label="Stats")],

    )

    demo.launch(server_name="127.0.0.1", server_port=49288)
    # auth=("admin", "pass1234")
