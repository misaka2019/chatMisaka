import sys
import os
import torch
import transformers
import torch.nn as nn
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModel


class ModifiedTrainer(transformers.Trainer):
    # 自定义compute_loss函数用于计算模型损失
    def compute_loss(self, model, inputs, return_outputs=False):
        # 调用model方法进行前向传播
        r = model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        )

        # 如果return_outputs为True，则返回损失和模型输出
        if return_outputs:
            return r.loss, r

        # 否则，只返回损失
        return r.loss

    # 自定义save_model函数用于保存训练好的模型
    def save_model(self, output_dir=None, _internal_call=False):
        # 导入TRAINING_ARGS_NAME常量
        from transformers.trainer import TRAINING_ARGS_NAME

        # 在指定目录下创建文件夹
        os.makedirs(output_dir, exist_ok=True)

        # 保存训练参数到TRAINING_ARGS_NAME文件中
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # 保存模型参数到adapter_model.bin文件中
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


def format_example(example):
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def train(
        # model/data params
        base_model='../hy-tmp/chatglm-6b',
        data_path="data/raw_data.json",
        output_dir='./lora-glm',
        # training hyperparams
        micro_batch_size=1,
        learning_rate=1e-4,
        cutoff_len=256,
        # lora hyperparams
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        # llm hyperparams
        train_on_inputs=True,  # if False, masks out inputs in loss
):
    device_map = "auto"
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    def tokenize(prompt, add_special_tokens=False):
        """
        将文本进行分词并编码为输入模型所需的格式

        Args:
            prompt (str): 待分词的文本
            add_special_tokens (bool, optional): 是否添加特殊标记（如[CLS]和[SEP]）。默认为False。

        Returns:
            dict: 分词后的文本，包括input_ids、attention_mask、token_type_ids、labels等字段。
        """
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            add_special_tokens=add_special_tokens
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        """
        对于每个对话数据点，生成完整的对话文本，并将其编码为输入模型所需的格式

        Args:
            data_point (dict): 包含对话上下文和对话文本的字典

        Returns:
            dict: 分词后的文本，包括input_ids、attention_mask、token_type_ids、labels等字段。
        """
        full_prompt = format_example(data_point)
        tokenized_full_prompt = tokenize(full_prompt["context"], True)
        tokenized_output = tokenize(full_prompt["target"])

        if not train_on_inputs:
            user_prompt_len = len(tokenized_full_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_output["labels"] + [
                tokenizer.eos_token_id]
        else:
            tokenized_full_prompt["labels"] = tokenized_full_prompt["labels"] + \
                                              tokenized_output["labels"] + [tokenizer.eos_token_id]
        tokenized_full_prompt["input_ids"] = tokenized_full_prompt["input_ids"] + \
                                             tokenized_output["input_ids"] + [tokenizer.eos_token_id]

        return tokenized_full_prompt

    data = load_dataset("json", data_files=data_path)

    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)

    model = AutoModel.from_pretrained(
        base_model,
        # load_in_8bit=True,
        # torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=device_map,
    )

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = False

    # model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        # target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.is_parallelizable = True
    model.model_parallel = True
    # peft_path = "../chatglm-lora/lora-glm/checkpoint-15000/adapter_model.bin"
    # model.load_state_dict(torch.load(peft_path), strict=False)
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=1,
            max_steps=50000,
            learning_rate=learning_rate,
            fp16=True,
            save_steps=1000,
            logging_steps=50,
            output_dir=output_dir,
            save_total_limit=5,
            group_by_length=False,
            seed=0,
            data_seed=0,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    train()
