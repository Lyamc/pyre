import torch,sys
from transformers import AutoTokenizer, AutoModelForCausalLM
d=sys.argv[1]; t=AutoTokenizer.from_pretrained(d); m=AutoModelForCausalLM.from_pretrained(d,torch_dtype=torch.float32)
p=t.apply_chat_template([{'role':'user','content':'Write a haiku about rust programming.'}],tokenize=False,add_generation_prompt=True)
ids=t(p,return_tensors='pt').input_ids
l=m(ids).logits[0,-1]; v,i=l.topk(5); print('top5',[(t.convert_ids_to_tokens(int(a)),int(a),round(float(b),3)) for a,b in zip(i,v)])
o=m.generate(ids,max_new_tokens=15,do_sample=False); print('greedy',o[0,ids.shape[1]:].tolist())
