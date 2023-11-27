
path = "/Users/relinda/VscodeWorkspace/codeLibrary/code/nlp_projects/LLM/ChatGLM-6B/ptuning/output/adgen-chatglm-6b-pt-128-2e-2/generated_predictions.txt"
with open(path, 'r', encoding='utf-8') as f:
  data1 = f.readlines()
  for line in data1:
    print(line.strip("\n"))
