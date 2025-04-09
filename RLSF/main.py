import os
import sys
import json
import math
import torch
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig
from trl.models import AutoModelForCausalLMWithValueHead
from peft import PeftModel
from lagent.llms import GPTAPI
from datetime import datetime
from lagent.actions import ActionExecutor, GoogleSearch
from lagent.llms import GPTAPI
from lagent.actions.GNews_API import ActionGNewsAPI
from lagent.actions.yahoo_finance import ActionYahooFinance
from Search.mindsearch.agent.mindsearch_agent import (
    MindSearchAgent,
    MindSearchProtocol,
    SearcherAgent,
)
from Search.mindsearch.agent.mindsearch_prompt import (
    FINAL_RESPONSE_CN,
    FINAL_RESPONSE_EN,
    GRAPH_PROMPT_CN,
    GRAPH_PROMPT_EN,
    searcher_context_template_cn,
    searcher_context_template_en,
    searcher_input_template_cn,
    searcher_input_template_en,
    searcher_system_prompt_cn,
    searcher_system_prompt_en,
    finance_system_prompt_cn,
    finance_system_prompt_en,
    News_system_prompt_cn,
    News_system_prompt_en,
)
import openai

lang = "cn"
# ============ 0. 分布式初始化 ============
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
world_size = dist.get_world_size()
rank = dist.get_rank()

torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# 如果想用 bitsandbytes 做 8bit/4bit 量化(可选)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float32
)


# ============ 1. 外部奖励模型类 ============
class ExternalRewardModel:
    def __init__(self, api_key, model="gpt-4o-mini"):
        self.model = "gpt-4o-mini"

    def get_reward(self, prompt, response):
        eval_prompt = (
            "你是一个AI评分助手。请评估以下回答的质量，给出0.01到0.99之间的分数。\n"
            "对于没有大问题的回答，给0.5以上的分数。\n"
            "你必须且只能回答一个数字，例如: 0.75。\n"
            "如果回答存在问题，请在0.01-0.50中按程度给分,不要给太多接近0.01的分数。\n"
            f"提示: {prompt}\n"
            f"回答: {response}\n\n"
            "评分（仅数字，无其他内容）:"
        )
        client = openai.OpenAI(
            api_key="",
            base_url="",
        )
        completion = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": eval_prompt}],
            max_tokens=5,
            temperature=0,
        )
        score_text = completion.choices[0].message.content
        try:
            score = float(score_text.strip())
            score = max(0.01, min(score, 0.99))
            # with open("score/1.txt", "a") as f:
            #     f.write(f"score: {score} ")
            reward = math.log(score / (1.0 - score))
            # with open("score/1.txt", "a") as f:
            #     f.write(f"reward: {reward}\n")
            return reward
        except Exception as e:
            print(f"Reward calculation failed: {e}, text: {score_text}")
            return 0.0


# ============ 2. 自定义数据集 ============
class MyCustomDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "query": item["input"],
            "instruction": item["instruction"],
            "output": item["output"],
        }


# ============ 3. 加载LoRA微调好的基础模型 ============
base_model_name_or_path = "Qwen/Qwen2.5-0.5B-Instruct"

# 使用 trl 的包装器加载基础模型
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    base_model_name_or_path
)
policy_model.to(device)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name_or_path, padding_side="left", trust_remote_code=True
)

# ============ 4. 仅在rank==0上加载外部奖励模型和MindSearch模块 ============
if rank == 0:
    reward_model = ExternalRewardModel("gpt-4o-mini")
    llm = GPTAPI(
        model_type="gpt-4o-mini",
        key="",
        openai_api_base="",
    )
    mindsearch_agent = MindSearchAgent(
        llm=llm,
        protocol=MindSearchProtocol(
            meta_prompt=datetime.now().strftime("The current date is %Y-%m-%d."),
            interpreter_prompt=GRAPH_PROMPT_CN if lang == "cn" else GRAPH_PROMPT_EN,
            response_prompt=FINAL_RESPONSE_CN if lang == "cn" else FINAL_RESPONSE_EN,
        ),
        searcher_cfg=dict(
            llm=llm,
            plugin_executor=ActionExecutor(
                GoogleSearch(api_key=""),
            ),
            protocol=MindSearchProtocol(
                meta_prompt=datetime.now().strftime("The current date is %Y-%m-%d."),
                plugin_prompt=(
                    searcher_system_prompt_cn
                    if lang == "cn"
                    else searcher_system_prompt_en
                ),
            ),
            template=dict(
                input=(
                    searcher_input_template_cn
                    if lang == "cn"
                    else searcher_input_template_en
                ),
                context=(
                    searcher_context_template_cn
                    if lang == "cn"
                    else searcher_context_template_en
                ),
            ),
        ),
        finance_searcher_cfg=dict(
            llm=llm,
            template=dict(
                input=(
                    searcher_input_template_cn
                    if lang == "cn"
                    else searcher_input_template_en
                ),
                context=(
                    searcher_context_template_cn
                    if lang == "cn"
                    else searcher_context_template_en
                ),
            ),
            plugin_executor=ActionExecutor(
                ActionYahooFinance(),
            ),
            protocol=MindSearchProtocol(
                meta_prompt=datetime.now().strftime("The current date is %Y-%m-%d."),
                plugin_prompt=(
                    finance_system_prompt_cn
                    if lang == "cn"
                    else finance_system_prompt_en
                ),
            ),
        ),
        news_searcher_cfg=dict(
            llm=llm,
            template=dict(
                input=(
                    searcher_input_template_cn
                    if lang == "cn"
                    else searcher_input_template_en
                ),
                context=(
                    searcher_context_template_cn
                    if lang == "cn"
                    else searcher_context_template_en
                ),
            ),
            plugin_executor=ActionExecutor(
                ActionGNewsAPI(api_key=""),
            ),
            protocol=MindSearchProtocol(
                meta_prompt=datetime.now().strftime("The current date is %Y-%m-%d."),
                plugin_prompt=(
                    News_system_prompt_cn if lang == "cn" else News_system_prompt_en
                ),
            ),
        ),
        max_turn=10,
    )
else:
    reward_model = None
    mindsearch_agent = None


# 定义调用MindSearch模块的函数
def get_mindsearch_answer(query, generated_response):
    # 组合输入，可根据需要调整格式或增加模板提示
    combined_input = query + "\n" + generated_response
    for agent_return in mindsearch_agent.stream_chat(combined_input):
        pass
    return agent_return.response


# ============ 5. PPO 配置与训练器 ============
ppo_config = PPOConfig(
    learning_rate=5e-6,
    batch_size=2,
    mini_batch_size=1,
    optimize_cuda_cache=True,
    max_grad_norm=0.5,
)
ppo_trainer = PPOTrainer(model=policy_model, config=ppo_config, tokenizer=tokenizer)

# ============ 6. 构造DataLoader + 分布式Sampler ============
data_path = "training.json"
dataset = MyCustomDataset(data_path)

train_sampler = DistributedSampler(
    dataset, num_replicas=world_size, rank=rank, shuffle=True
)
train_dataloader = DataLoader(
    dataset, batch_size=2, shuffle=False, sampler=train_sampler, drop_last=True
)

num_epochs = 2

# ============ 7. 训练循环 ============
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    if rank == 0:
        print(f"Epoch {epoch+1}/{num_epochs}...")

    for batch_data in train_dataloader:
        queries = batch_data["query"]

        # 1) 各进程生成回答
        query_tensors = tokenizer(
            queries, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).input_ids.to(device)
        # 此处转换为list形式，确保每个样本是单独tensor
        query_tensors = [torch.tensor(q).to(device) for q in query_tensors]

        responses_tensors = ppo_trainer.generate(
            query_tensors,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
        )
        responses = tokenizer.batch_decode(responses_tensors, skip_special_tokens=True)

        # 2) 各进程将query/response拼接后发送至rank==0计算reward
        local_str_list = [q + "\n" + r for q, r in zip(queries, responses)]
        joined_str = "<sep>".join(local_str_list)
        joined_bytes = joined_str.encode("utf-8")
        local_size = torch.tensor([len(joined_bytes)], dtype=torch.long, device=device)

        sizes_list = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(sizes_list, local_size)

        max_size = max(s.item() for s in sizes_list)
        padded = torch.zeros((max_size,), dtype=torch.uint8, device=device)
        padded[: local_size.item()] = torch.tensor(
            list(joined_bytes), dtype=torch.uint8, device=device
        )

        gathered_list = [
            torch.zeros((max_size,), dtype=torch.uint8, device=device)
            for _ in range(world_size)
        ]
        dist.all_gather(gathered_list, padded)

        all_rewards = []
        if rank == 0 and reward_model is not None:
            for i, size_tensor in enumerate(sizes_list):
                raw_size = size_tensor.item()
                raw_bytes = gathered_list[i][:raw_size].cpu().numpy().tobytes()
                raw_str = raw_bytes.decode("utf-8")
                pairs = raw_str.split("<sep>")
                for pair in pairs:
                    q_r = pair.split("\n", 1)
                    if len(q_r) == 2:
                        q_, r_ = q_r
                        try:
                            mindsearch_answer = get_mindsearch_answer(q_, r_)
                        except Exception as e:
                            print(f"MindSearch调用失败: {e}")
                            mindsearch_answer = r_  # 出错则使用原回答
                        rw = reward_model.get_reward(q_, mindsearch_answer)
                        all_rewards.append(rw)
        # 3) 将rank==0计算的奖励广播回所有进程
        count = torch.tensor([len(all_rewards)], dtype=torch.long, device=device)
        dist.broadcast(count, src=0)
        if rank != 0:
            all_rewards = [0.0] * count.item()

        if count.item() > 0:
            reward_tensor = torch.tensor(all_rewards, dtype=torch.float, device=device)
            dist.broadcast(reward_tensor, src=0)
            all_rewards = reward_tensor.tolist()

        query_lens = torch.tensor([len(queries)], device=device)
        query_lens_list = [torch.zeros_like(query_lens) for _ in range(world_size)]
        dist.all_gather(query_lens_list, query_lens)
        start_idx = sum([query_lens_list[i].item() for i in range(rank)])
        end_idx = start_idx + len(queries)
        current_rewards = all_rewards[start_idx:end_idx]

        # 4) PPO更新（仅针对当前进程样本）
        query_tensors_step = tokenizer(queries, padding=True, return_tensors="pt").to(
            device
        )
        response_tensors_step = tokenizer(
            responses, padding=True, return_tensors="pt"
        ).to(device)

        query_list = [
            query_tensors_step["input_ids"][i]
            for i in range(query_tensors_step["input_ids"].size(0))
        ]
        response_list = [
            response_tensors_step["input_ids"][i]
            for i in range(response_tensors_step["input_ids"].size(0))
        ]

        if len(query_list) == ppo_config.batch_size:
            reward_tensors = [torch.tensor([r], device=device) for r in current_rewards]
            assert all(
                torch.isfinite(r) for r in reward_tensors
            ), "Rewards中包含inf或nan!"

            stats = ppo_trainer.step(query_list, response_list, reward_tensors)
            print(f"[rank={rank}] PPO stats: {stats}")
        else:
            print(
                f"[rank={rank}] Batch size mismatch. Expected {ppo_config.batch_size}, got {len(query_list)}. Skipping."
            )

# ============ 8. 保存模型（仅在rank==0保存） ============
if rank == 0:
    save_path = "/saves/ppo_finetuned_lora_model"
    if hasattr(ppo_trainer.model, "module"):
        ppo_trainer.model.module.save_pretrained(save_path)
    else:
        ppo_trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"[rank=0] Model and tokenizer saved to {save_path}")

dist.barrier()
dist.destroy_process_group()

# 运行示例：
# CUDA_LAUNCH_BLOCKING=1 TORCH_DISTRIBUTED_DEBUG=INFO torchrun --standalone --nproc_per_node=8 main.py
