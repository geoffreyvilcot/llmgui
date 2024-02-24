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
    in_data = {"prompt": prompt, "n_predict": max_tokens, "stop" : json_stop, "seed": seed}

    # api_url = f"{url}/embedding"
    # in_data = {"content": prompt}

    headers = {"Content-Type": "application/json"}
    print(f"sending : {in_data}")
    response = requests.post(api_url, data=json.dumps(in_data), headers=headers)


    print(str(response.text))
    jstring = json.loads(response.text)

    answer = jstring['content'] #.encode('utf-8',errors='ignore')

    stats = f"prompt tokens per second: {jstring['timings']['prompt_per_second']} ; predicted tokens per second: {jstring['timings']['predicted_per_second']} "

    return answer, stats


if __name__ == "__main__":

    demo = gr.Interface(
        allow_flagging='never',
        fn=query,
        inputs=[gr.Textbox(label="Url", value="http://127.0.0.1:8080"),
                gr.Textbox(label="Pre Prompt", lines=5),
                gr.Textbox(label="Inputs", lines=10),
                gr.Textbox(label="Stop work"),
                gr.Number(512, label="Max tokens"),
                gr.Number(-1, label="Seed"),
                ],
        outputs=[gr.Textbox(label="Outputs", lines=30), gr.Label(label="Stats")],

    )

    demo.launch(server_name="127.0.0.1", server_port=49288)
    # auth=("admin", "pass1234")
