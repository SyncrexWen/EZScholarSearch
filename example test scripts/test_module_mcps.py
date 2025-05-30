# flake8: noqa

from ezscholarsearch import mcps
from ezscholarsearch import AI as ai
from functools import reduce

import os

# To be tested
# Model, finished, passed
# Skill, finished, passed
# FunctionSkill, finished, passed
# Agent, finished, passed
# MCP, same to 'Agent' module

BASE_URL = "base_url"
API_KEY = "<Your API Key>"
MODEL = "<MODEL>"


config = {
    "base_url": BASE_URL,
    "api_key": API_KEY,
    "model": MODEL
}

saving_dir = "test_tmp_files"
os.makedirs(saving_dir, exist_ok=True)

with open("test_text.txt", 'r', encoding='utf-8') as f:
    text1 = f.read()

def save_output(data: str, name: str, saving_dir: str = saving_dir) -> None:
    if not data:
        return
    path = os.path.join(saving_dir, name + ".md")
    with open(path, 'w', encoding='utf-8') as file:
        file.write(str(data))

def test_Model():

    model = mcps.Model(
        name="translator",
        role="以相关专业人员的视角翻译输入文字为中文",
        **config
    )

    output = model(text1)

    assert isinstance(output, mcps.DataPacket)
    save_output(output.to_str(), "Model-Call-文字翻译")

    save_output(
        data=str(model),
        name="Model-strf-字符化"
    )

def test_Skill():

    class MySkill(mcps.Skill):
        def __init__(self, name, role, *funcs: callable, **calls: callable):
            super().__init__(name, role)
            self.calls = list(funcs) + list(calls.values())

        def call(self, data: mcps.DataPacket):
            return reduce(lambda acc, func: func(acc), self.calls, data)
        
    def iadd(i=1):
        def wrapper(data: mcps.DataPacket):
            if not data.validate(int):
                return mcps.DataPacket(content=0)
            else:
                return mcps.DataPacket(content=data.content + i)
        return wrapper

    iadd_skill = MySkill(
        "adder",
        "operate a sequetial adding operation",
        iadd(1),
        iadd(2),
        iadd(3),
        my_iadd=iadd(-10),
        my_add=iadd(114514)
    )

    output = iadd_skill("Test Script")
    assert isinstance(output, mcps.DataPacket)
    assert output.validate(int)
    assert output.content == 2 + 3 - 10 + 114514

def test_FunctionSkill():
    def my_add(data: mcps.DataPacket):
        if data.validate(int):
            return mcps.DataPacket(content=data.content + 1314520)
        elif data.validate():
            return mcps.DataPacket(content=114514)
        else:
            return mcps.DataPacket(content=1919810)
        
    func = mcps.FunctionSkill(
        "my_add",
        "operate my_add",
        my_add
    )

    output1 = func("123")
    output2 = func(123)
    output3 = func([])

    def check(data):
        assert isinstance(data, mcps.DataPacket)
        assert data.validate(int)

    check(output1)
    assert output1.content == 114514
    check(output2)
    assert output2.content == 123 + 1314520
    check(output3)
    assert output3.content == 1919810

def test_agent():
    agent = mcps.Agent(
        planner_mode='llm',
        aggregator_mode='md',
        **config
    )

    agent.register(
        "summerizer",
        "总结中文文段",
        mcps.Model(
            name='summerizer',
            role='总结中文文段',
            system_prompt="你是一个文段总结专家，请简明扼要地总结给你的文段",
            **config,
        )
    )

    fct = ai.AIModelFactory(
        **config
    )
    agent.register(
        name='translator',
        role='将文段翻译为中文',
        tool=fct("请以相关专业从业者的视角将给你的文字翻译为中文")
    )

    tool = ai.ToolBuilder(
        name='introduce_law',
        description="讲解公式的具体内容、发现者与历史"
    )
    tool.add_param(
        name="content",
        description="公式的内容",
        param_type='string'
    )
    tool.add_param(
        name="discoverer",
        description="讲解公式的发现者",
    )
    tool.add_param(
        name='history',
        description="讲解公式发现的历史"
    )
    agent.register(
        name='law_introduce',
        role='介绍公式',
        tool=fct(
            "你是一个乐于助人的人工智能助手，你会介绍公式的内容、发现者和历史",
            tools=[tool.build()],
            tool_choice='required',
        )
    )
    movie_data = "<Harry Potter>, 114514"
    agent.register(
        name='data_getter',
        role='从本地观影数据集中获得观影数据',
        tool=mcps.FunctionSkill(
            name='data_getter',
            role='从本地数据集中获得观影数据',
            func=lambda data: mcps.DataPacket(movie_data)
        )
    )

    translate_output = agent("翻译下面这段文字至中文：\n" + text1)
    assert isinstance(translate_output, mcps.DataPacket)
    save_output(translate_output.to_str(), "Agent-Model-Call-翻译")

    summerize_output = agent("总结下面的文字:\n" + translate_output.to_str())
    assert isinstance(summerize_output, mcps.DataPacket)
    save_output(summerize_output.to_str(), "Agent-AIModel-Call-总结")

    introduction_output = agent("为我介绍麦克斯韦方程组的具体公式")
    assert isinstance(introduction_output, mcps.DataPacket)
    save_output(introduction_output.to_str(), "Agent-AIModel-ToolsCall-介绍公式")

    data_getter_output = agent("请获取储存在本地的观影数据")
    assert isinstance(data_getter_output, mcps.DataPacket)
    save_output(data_getter_output.to_str(), 'Agent-FunctionSkill-本地数据获取')
