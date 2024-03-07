import json

class Config(object):
    def __init__(self, conf_file="config.json"):
        with open(conf_file, "rt", encoding="utf8") as f :
            jconf = json.load(f)

        self.prompt_template = jconf['prompt_template']
        self.user_name = jconf['user_name']
        self.bot_name = jconf['bot_name']

        if "listen_bind" in jconf :
            self.listen_bind = jconf['listen_bind']
        else :
            self.listen_bind = "127.0.0.1"

        if "listen_port" in jconf:
            self.listen_port = int(jconf['listen_port'])
        else:
            self.listen_port = 49283

        if "external_llama_cpp_url" in jconf and len(jconf['external_llama_cpp_url']) > 5:
            self.external_llama_cpp_url = jconf['external_llama_cpp_url']
        else :
            self.external_llama_cpp_url = None

        if "external_llama_cpp_api_key" in jconf and len(jconf['external_llama_cpp_api_key']) > 0:
            self.external_llama_cpp_api_key = jconf['external_llama_cpp_api_key']
        else :
            self.external_llama_cpp_api_key = None


