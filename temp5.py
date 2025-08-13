from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models import GptOssForCausalLM

model = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    model,
    torch_dtype="bfloat16",
    device_map="cuda",
)

model.eval()
input_ids = tokenizer("hi there", return_tensors="pt").input_ids.cuda()

print(model(input_ids))

output = model.generate(
    input_ids,
    max_new_tokens=10,
    do_sample=False,
    temperature=0.0,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
