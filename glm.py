from transformers import AutoModel, AutoTokenizer
import torch
from peft import get_peft_model, LoraConfig, TaskType

lora_weights = "../chatglm-lora/lora-glm/adapter_model.bin"
tokenizer = AutoTokenizer.from_pretrained("./model", trust_remote_code=True)
model = AutoModel.from_pretrained("./model", trust_remote_code=True, load_in_8bit=True, device_map='auto')

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=True,
    r=8,
    lora_alpha=32, lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(lora_weights), strict=False)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
model = model.eval()


def format_example(example):
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def predict(instruction, top_p=0.7, temperature=0.1):
    with torch.no_grad():
        feature = format_example({'instruction': f'{instruction}', 'output': '', 'input': ''})
        input_text = feature['context']
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        out = model.generate(
            input_ids=input_ids,
            max_length=256,
            top_p=top_p,
            temperature=temperature,
        )
        answer = tokenizer.decode(out[0])
        return answer.split(':')[1]
