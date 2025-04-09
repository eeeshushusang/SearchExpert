from datetime import datetime
from lagent.actions import ActionExecutor, GoogleSearch
from lagent.llms import GPTAPI
from mindsearch.agent.mindsearch_agent import (
    MindSearchAgent,
    MindSearchProtocol,
    SearcherAgent,
)
from mindsearch.agent.mindsearch_prompt import (
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
from lagent.actions.GNews_API import ActionGNewsAPI
from lagent.actions.yahoo_finance import ActionYahooFinance
from config import API_KEYS
from tqdm import trange
import time
import re

lang = "cn"
# llm = GPTAPI(model_type="llama3.1-70b", key=API_KEYS['llama_api'],openai_api_base="")
# llm = GPTAPI(model_type="gpt-4o-mini", key=API_KEYS['claude'],openai_api_base="")
# llm = GPTAPI(model_type="claude-3-5-sonnet-20240620", key=API_KEYS['claude'],openai_api_base="")
# llm = GPTAPI(model_type="claude-3-haiku", key=API_KEYS['claude'],openai_api_base="")
# llm = GPTAPI(model_type="gemini-1.5-flash-latest", key=API_KEYS['claude'],openai_api_base="")
llm = GPTAPI(model_type="gpt-4o-mini", key=API_KEYS["gpt"], openai_api_base="")
# llm = GPTAPI(model_type="deepseek-chat",key=API_KEYS['deepseek'],openai_api_base="")


agent = MindSearchAgent(
    llm=llm,
    protocol=MindSearchProtocol(
        meta_prompt=datetime.now().strftime("The current date is %Y-%m-%d."),
        interpreter_prompt=GRAPH_PROMPT_CN if lang == "cn" else GRAPH_PROMPT_EN,
        response_prompt=FINAL_RESPONSE_CN if lang == "cn" else FINAL_RESPONSE_EN,
    ),
    searcher_cfg=dict(
        llm=llm,
        plugin_executor=ActionExecutor(
            GoogleSearch(api_key=API_KEYS["google_search"]),
        ),
        protocol=MindSearchProtocol(
            meta_prompt=datetime.now().strftime("The current date is %Y-%m-%d."),
            plugin_prompt=(
                searcher_system_prompt_cn if lang == "cn" else searcher_system_prompt_en
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
                finance_system_prompt_cn if lang == "cn" else finance_system_prompt_en
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
            ActionGNewsAPI(api_key=API_KEYS["gnews"]),
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

# 初始化数组 A[0..1500]
A = [None] * 1501
A[
    0
] = """根据以下节点和边的描述.生成代码并执行,获得最终结果。
注意WebSearchGraph类已经存在,请不要自己定义.
root节点名称固定为root.不要改变.
耐心等待代码执行
最后回答A,B,C,D,E这样的选项.
"""

# 打开并读取文本文件
with open("assets/4.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

j = 1
# 遍历所有行
for line in lines:
    text = line.strip()
    A[j] = text
    j = j + 1

# print(A[1])
# print(A[1500])

# 打开一个文件用于写入答案
with open("assets/llama.txt", "a", encoding="utf-8") as f:
    for i in trange(1, 101, desc="Question"):
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                for agent_return in agent.stream_chat(A[0] + "\n" + A[i]):
                    pass
                answer = agent_return.response
                f.write(A[i] + "\n")
                f.write(f"{i}. {answer}\n" + "\n")
                print("Success")
                break  # 成功，跳出重试循环
            except Exception as e:
                print(f"第 {i} 题处理出现错误: {e}，尝试第 {attempt} 次。")
                if attempt < max_retries:
                    wait_time = attempt
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    f.write(f"第 {i} 题处理失败，跳过。")
