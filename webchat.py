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

def query(message, history, preprompt, user_name : str, bot_name : str, max_tokens, seed):

    history_str = ""
    for h in history :
        sub = "\n".join(h)
        history_str += f"{sub}"

    prompt = f"{preprompt}\n{history_str}\n{message}\n{bot_name}:"
    # prompt = f"{history_str}\n{message}"

    print(prompt)

    # system_prompt = {
    #         "prompt": f"Transcript of a never ending dialog, where the {user_name} interacts with an {bot_name}.\nThe {bot_name} is helpful, kind, honest, good at writing, and never fails to answer the {user_name}'s requests immediately and with precision.\n{user_name}: Recommend a nice restaurant in the area.\n{bot_name}: I recommend the restaurant \"The Golden Duck\". It is a 5 star restaurant with a great view of the city. The food is delicious and the service is excellent. The prices are reasonable and the portions are generous. The restaurant is located at 123 Main Street, New York, NY 10001. The phone number is (212) 555-1234. The hours are Monday through Friday from 11:00 am to 10:00 pm. The restaurant is closed on Saturdays and Sundays.\n{user_name}: Who is Richard Feynman?\n{bot_name}: Richard Feynman was an American physicist who is best known for his work in quantum mechanics and particle physics. He was awarded the Nobel Prize in Physics in 1965 for his contributions to the development of quantum electrodynamics. He was a popular lecturer and author, and he wrote several books, including \"Surely You're Joking, Mr. Feynman!\" and \"What Do You Care What Other People Think?\".\n{user_name}:",
    #         "anti_prompt": f"{user_name}:",
    #         "assistant_name": f"{bot_name}:"
    # }
    start_t = time.time()
    api_url = f"{conf.external_llama_cpp_url}/completion"
    in_data = {"prompt": prompt, "n_predict": max_tokens, "stop" : [f"{user_name}:"], "seed": seed}
    # in_data = {"prompt": prompt, "n_predict": max_tokens, "system_prompt" : system_prompt, "seed": seed}

    # {
    #     "system_prompt": {
    #         "prompt": "Transcript of a never ending dialog, where the User interacts with an Assistant.\nThe Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.\nUser: Recommend a nice restaurant in the area.\nAssistant: I recommend the restaurant \"The Golden Duck\". It is a 5 star restaurant with a great view of the city. The food is delicious and the service is excellent. The prices are reasonable and the portions are generous. The restaurant is located at 123 Main Street, New York, NY 10001. The phone number is (212) 555-1234. The hours are Monday through Friday from 11:00 am to 10:00 pm. The restaurant is closed on Saturdays and Sundays.\nUser: Who is Richard Feynman?\nAssistant: Richard Feynman was an American physicist who is best known for his work in quantum mechanics and particle physics. He was awarded the Nobel Prize in Physics in 1965 for his contributions to the development of quantum electrodynamics. He was a popular lecturer and author, and he wrote several books, including \"Surely You're Joking, Mr. Feynman!\" and \"What Do You Care What Other People Think?\".\nUser:",
    #         "anti_prompt": "User:",
    #         "assistant_name": "Assistant:"
    #     }
    # }

    headers = {"Content-Type": "application/json"}

    print(f"sending : {in_data}")
    response = requests.post(api_url, data=json.dumps(in_data), headers=headers)


    print(str(response.text))
    jstring = json.loads(response.text)

    answer = jstring['content'] #.encode('utf-8',errors='ignore')

    stats = f"prompt tokens per second: {jstring['timings']['prompt_per_second']} ; predicted tokens per second: {jstring['timings']['predicted_per_second']} "

    return answer

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
    with open(conf.prompt_template, "r", encoding='utf-8') as f:
        pre_prompt = f.read()

#     pre_prompt = """This is a transcript of a dialog between User and Llama.
# Llama is a friendly chatbot with a huge knowledge.
# Llama is honest, answer with exactitude and precision."""

    gr.ChatInterface(query,
                     additional_inputs=[
                         # gr.Textbox(label="Url", value=conf.external_llama_cpp_url),
                         gr.Textbox(label="Pre Prompt", lines=5, value=pre_prompt),
                         gr.Textbox(label="User name", value=conf.user_name),
                         gr.Textbox(label="Bot name", value=conf.bot_name),
                         gr.Number(512, label="Max tokens"),
                         gr.Number(-1, label="Seed"),
                     ]
                     ).launch(server_name=conf.listen_bind, server_port=conf.listen_port)