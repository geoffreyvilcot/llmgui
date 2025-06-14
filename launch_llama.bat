cd G:\bidule_llama\llama-b5630-bin-win-cuda-12.4-x64

rem llama-server.exe --jinja --ctx-size 40960 -m g:\bidule_llama\models\Magistral-Small-2506-Q4_K_M.gguf -ngl 99

llama-server.exe --jinja --ctx-size 8096 -m g:\bidule_llama\models\Magistral-Small-2506-Q4_K_M.gguf -ngl 99