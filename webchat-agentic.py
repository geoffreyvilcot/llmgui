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
import base64
import re
import asyncio
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright

playwright_browser = None

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

def extract_json_objects_fara(llm_response):
    # Regex to capture the JSON object between <tool_call> tags
    
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', llm_response, re.DOTALL)
    if match:
        json_str = match.group(1)
        data = json.loads(json_str)
        print("Extracted JSON:", data)
        return data, llm_response.split('<tool_call>')[0]
    else:
        return None, llm_response.split('<tool_call>')[0]

def extract_json_objects_mistral(llm_response):
    # Regex to capture the JSON object between <tool_call> tags
    # llm_response = llm_response.replace('"name":', '"action":')
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', llm_response, re.DOTALL)
    if match:
        json_str = match.group(1)
        data = json.loads(json_str)
        # data['arguments'] = {"action": data['action']}
        print("Extracted JSON:", data)
        return data, llm_response.split('<tool_call>')[0]
    else:
        return None, llm_response.split('<tool_call>')[0]

def action_type(page, coordinate, text, press_enter=False):
    page.mouse.click(coordinate[0], coordinate[1])
    page.keyboard.type(text)
    if press_enter:
        page.keyboard.press("Enter")
 
def query(message, history, systemprompt, max_tokens, seed, temp, top_p, max_iteration):
    gradio_response = ""
    messages = []
    messages.append({"role": "system", "content": systemprompt})
    messages = messages + history

    start_url = "http://localhost:3000/"

    # messages.append({"role": "user", "content": message})
    # llm_response = call_llm_api_v1(messages, max_tokens, seed, temp, top_p)    
    # data, text = extract_json_objects_fara(llm_response)

    # gradio_response += text
    # yield gradio_response

    # if data :
    #     if data['arguments']['action'] == 'visit_url':
    #         start_url = data['arguments']['url']
    #         if not start_url.startswith("http"):
    #             start_url = "https://" + start_url
    #         print(f"[INFO] visiting {start_url}")
    #     else :
    #         print("[ERROR] no visit_url action found, using default start_url")
    #         return("Error: no visit_url action found in LLM response.")
    # else :
    #     print("[ERROR] no JSON found between <tool_call> tags.")
    #     return("Parse error: No JSON found between <tool_call> tags.")

    # messages.append({"role": "assistant", "content": llm_response})
    user_content = [{"type": "text", "text": message}]
    messages.append({"role": "user", "content": user_content})

    llm_response = call_llm_api_v1(messages, max_tokens, seed, temp, top_p)   

    p, browser, page = init_agent()
    page.goto(start_url, wait_until='domcontentloaded')

    for step_idx in range(max_iteration):  # limite nombre d'étapes pour éviter boucles infinies
        time.sleep(2)  # attendre que la page charge
        # 1) récupérer le contenu & screenshot
        page_text = page.content()
        screenshot = page.screenshot(scale = "css")  # bytes

        # include a short base64 preview if available (truncated)
        message = f"{page_text[:4000]}\n\n"
        b64 = base64.b64encode(screenshot).decode()
        # message += f"Screenshot (base64, truncated): {b64[:2000]}\n\n"
        # messages.append({"role": "user", "content": message})
        user_content = [{"type": "text", "text": message}]
        if b64: 
            user_content += [{"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}}]
        messages.append({"role": "user", "content": user_content})

        llm_response = call_llm_api_v1(messages, max_tokens, seed, temp, top_p)       
        
        # Regex to capture the JSON object between <tool_call> tags
        data, text = extract_json_objects_fara(llm_response)
        gradio_response += text
        yield gradio_response

        if data:
            if data['arguments']['action'] == 'terminate':
                print("[INFO] terminating agent as per LLM instruction.")
                gradio_response += "\nAgent terminated as per LLM instruction."
                yield gradio_response
                return gradio_response
            if data['arguments']['action'] == 'type':
                coord = data['arguments']['coordinate']
                text = data['arguments']['text']
                press_enter = data['arguments']['press_enter']
                print(f"[INFO] typing '{text}' at {coord}, press_enter={press_enter}")
                action_type(page, coord, text, press_enter)
            if data['arguments']['action'] == 'left_click':
                coord = data['arguments']['coordinate']
                page.mouse.click(coord[0], coord[1])
                print(f"[INFO] left clicked at {coord}")
            if data['arguments']['action'] == 'scroll':
                direction = data['arguments']['direction']
                if direction == 'down':
                    page.keyboard.press("PageDown")
                elif direction == 'up':
                    page.keyboard.press("PageUp")
                print(f"[INFO] scrolled {direction}")

        else:
            return("Parse error: No JSON found between <tool_call> tags.")

 
        # browser.close()
    print("[INFO] fin des étapes de l'agent.")
    gradio_response += "\nfin des étapes de l'agent."
    yield gradio_response
    return gradio_response

def call_llm_api(messages, max_tokens, seed, temp, top_p):
    headers = {"Content-Type": "application/json"}
    in_data = {"messages": messages}

    api_url = f"{conf.external_llama_cpp_url}/apply-template"
    response = requests.post(api_url, data=json.dumps(in_data), headers=headers)

    prompt = json.loads(response.content)["prompt"]
    start_t = time.time()
    api_url = f"{conf.external_llama_cpp_url}/completion"
    in_data = {"prompt": prompt, "n_predict": max_tokens, "seed": seed, "stream" : False, "temperature" : temp, "top_p" : top_p}
    # print(f"sending to LLM API: {in_data}")
    response = requests.post(api_url, data=json.dumps(in_data), headers=headers, stream=False)

    print(response)
    
    print(json.loads(response.text)['content'])
    return(json.loads(response.text)['content'])

def call_llm_api_v1(messages, max_tokens, seed, temp, top_p):

    start_t = time.time()
    api_url = f"{conf.external_llama_cpp_url}/v1/chat/completions"
    in_data = {"model": "ministral", "messages": messages, "n_predict": max_tokens, "seed": seed, "stream" : False, "temperature" : temp, "top_p" : top_p}
    # print(f"sending to LLM API: {in_data}")
    response = requests.post(api_url, data=json.dumps(in_data), stream=False)    
    msg_response = json.loads(response.text)['choices'][0]['message']
    return(msg_response['content'])


def init_agent():
    global playwright_browser
    if not playwright_browser :
        playwright_browser =  sync_playwright().start() 
    browser = playwright_browser.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    return playwright_browser, browser, page
    # await page.goto("https://www.example.com")
    # content = await page.content()
    # print(content)
    # await browser.close()


if __name__ == "__main__":

    conf_file_name = "config_fara.json"

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
    
    messages = []
    messages.append({"role": "system", "content": "You are an assistant that helps users to plan their trips in France."})
    messages.append({"role": "user", "content": "Bonjour, Calcul un itinéraire entre Tours et Orléans"})
    # r = call_llm_api_v1(messages, 2048, -1, 0.7, 0.95)
    # print(r)

    # p, browser, page = init_agent()

    gr.ChatInterface(query,
                     analytics_enabled=False,
                     additional_inputs=[
                         gr.Textbox(label="System Prompt", lines=5, value=system_prompt),
                         gr.Number(20480, label="Max tokens"),
                         gr.Number(-1, label="Seed"),
                         gr.Number(0.7, label="temp"),
                         gr.Number(0.95, label="top_p"),
                         gr.Number(10, label="Max iteration"),
                     ]
                     ).launch(server_name=conf.listen_bind, server_port=conf.listen_port, root_path=conf.root_path)

