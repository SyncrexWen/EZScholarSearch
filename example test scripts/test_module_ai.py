# flake8: noqa: F401

from ezscholarsearch import AI as ai

from collections import Counter
from time import sleep
import os
import re

# To be tested
# 'AIModel',    Finished, passed
# 'AIModelFactory', Finished, passed
# 'WorkFlow',   Finished, passed
# 'DataProcessor',  Finished, passed
# 'SequentialBlock',    Finished, passed
# 'ParallelBlock',  Finished, passed
# 'SequenceProcessor', Finished, passed
# 'MultiThreadsSequenceProcessor', Finished, passed


BASE_URL = "base_url"
API_KEY = "<Your API Key>"
MODEL = "<model>"

with open("test_text.txt", 'r', encoding='utf-8') as file:
    text = file.read()


with open("test_text2.txt", 'r', encoding='utf-8') as file:
    text2 = file.read()

saving_dir = "test_tmp_files"
os.makedirs(saving_dir, exist_ok=True)


def save_md(title: str, content: str, saving_dir=saving_dir):
    path = os.path.join(saving_dir, title + ".md")
    with open(path, 'w', encoding='utf-8') as file:
        file.write(f"# {title}\n---\n")
        file.write(content)
    return content


def save_json(title: str, content: dict, saving_dir=saving_dir):
    path = os.path.join(saving_dir, title + ".md")
    with open(path, 'w', encoding='utf-8') as file:
        file.write(f"# {title}\n\n")
        file.write("\n---\n".join([
            f"## {key}\n\n{value}"
            for key, value in content.items()
        ]))
    return content


def test_AIModel_SystemPrompt():
    prompt = "接下来我会给你一些定理，请分别简述他们的内容、发明者、发现的历史、关键的论文简述"
    fct = ai.AIModelFactory(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
    )

    model = fct(prompt, temperature=0.3, output_type='str')
    assert isinstance(model, ai.AIModel)

    response1 = model("热力学三定律")

    assert isinstance(response1, str)
    assert response1
    save_md("AIModel-SystemPrompt-热力学三定律简述", response1)

    response2 = model("热力学三定律", output_type='default')
    assert isinstance(response2, ai.DataPacket)
    assert response2.content


def test_AIModel_tools_call():
    prompt = "接下来我会给你一些定理，请分别简述他们的内容、发明者、发现的历史"
    fct = ai.AIModelFactory(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL
    )

    model = fct(
        system_prompt=prompt,
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "introduce_law",
                    "description": "用中文介绍公式的内容、发明者、发现的历史",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "公式的内容"
                            },
                            "discoverer": {
                                "type": "string",
                                "description": "公式的发现者"
                            },
                            "history": {
                                "type": "string",
                                "description": "公式发现的历史"
                            }
                        },
                        "required": ["content", "discoverer", "history"]
                    }
                }
            }
        ],
        tool_choice='auto',
    )

    tool_call = model("热力学三定律", output_type='default')
    assert isinstance(tool_call, ai.DataPacket)
    assert tool_call.metadata

    save_json("AIModel-ToolCall-热力学三定律", tool_call.metadata)
    save_md(
        "AIModel-ToolCall-热力学三定律",
        "\n---\n".join([
            f"# {key}\n\n{value}"
            for key, value in tool_call.metadata.items()
        ])
    )


def test_DataProcessor():
    def cut(text: str):
        words = re.findall(r"\b\w+\b", text)
        cnt = dict(sorted(Counter(words).items(), key=lambda x: x[1]))
        return cnt

    processor = ai.DataProcessor(lambda x: ai.DataPacket(content=None, metadata=cut(x.content)))

    response = processor(text)
    assert isinstance(response, ai.DataPacket)
    assert response.metadata and not response.content

    save_md("DataProcessor-词频计数",
        "\n".join([f"{word}: {count}" for word, count in response.metadata.items()]))


def test_SequentialBlock():
    fct = ai.AIModelFactory(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
    )

    sequential_block = ai.SequentialBlock(
        fct("请以生物医药专业科研从业者的视角翻译英文文章至中文", temperature=0.4),
        ai.DataProcessor(lambda x: ai.DataPacket(content=save_md("SequentialBlock-翻译", x.content))),
        fct("请以生物医药专业科研从业者的视角总结这段论文的摘要", temperature=1.5)
    )

    response = sequential_block(text)
    assert isinstance(response, ai.DataPacket)
    assert response.content and not response.metadata

    save_md("SequentialBlock-总结", response.content)


def test_ParallelBlock_ParallelInput():
    fct = ai.AIModelFactory(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL
    )

    parallel_block = ai.ParallelBlock(
        translate=fct("请以生物医药专业科研从业者的视角翻译英文文章至中文", temperature=0.4),
        summerize=fct("Summerize this essay abstract from the perspective of a scientific research practitioners of biomedicine")
    )

    response = parallel_block(text)
    assert isinstance(response, ai.DataPacket)
    assert not response.content and response.metadata

    save_json("ParallelBlock-ParallelInput-解析文献", response.metadata)


def test_ParallelBlock_KeyWordInput():
    fct = ai.AIModelFactory(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL
    )

    input_data = ai.DataPacket(
        content=None,
        metadata={
            "translate1": text,
            "translate2": text2,
        }
    )

    parallel_block = ai.ParallelBlock(
        translate1=fct("请以生物医药专业科研从业者的视角翻译英文文章至中文", temperature=0.4),
        translate2=fct("请以生物医药专业科研从业者的视角翻译英文文章至中文", temperature=0.4)
    )

    response = parallel_block(input_data)
    assert isinstance(response, ai.DataPacket)
    assert not response.content and response.metadata

    save_json("ParallelBlock-KeyWordInput-文献翻译", response.metadata)


def test_sequence_processor():
    words = re.findall(r"\bw+\b", text)
    def task(element):
        try:
            hash(element)
            sleep(0.1)
            return True
        except:
            return False

    sequence_processor = ai.SequenceProcessor(task)
    multi_treads_sequence_processor = ai.MultiThreadsSequenceProcessor(task, max_workers=5)

    result1 = sequence_processor(words)

    result2 = multi_treads_sequence_processor(words)

    assert result1 == result2


def test_WorkFlow():
    fct = ai.AIModelFactory(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL
    )

    def save_datapacket_output(title: str):
        def wrapper(datapacket: ai.DataPacket):
            if datapacket.content:
                save_md(title, datapacket.content)
            elif datapacket.metadata:
                save_json(title, datapacket.metadata)
            return datapacket
        return wrapper

    class MyWorkFlow(ai.WorkFlow):
        def __init__(self):

            self.search = fct('为我寻找这个作者的诗，输出仅保留标题和内容，注意千万不要出现作者名称', temperature=1.4)

            self.save_search = ai.DataProcessor(save_datapacket_output("WorkFlow-诗歌搜索"))

            self.feature = ai.ParallelBlock(
                summerize=fct("请简要总结这些诗歌的风格，注意不要保留任何作者和作品名称", temperature=1.4),
                translate=fct('请将这些诗歌有诗意地翻译成英文', temperature=1.8),
                content=fct("接下来我会给你一些诗歌，我需要你完成下面的任务：我们要进行一个游戏，描述一个诗人诗歌的内容、诗人的人生态度等，让别人猜诗人是谁。"
                            "现在请你进行出题，注意不能暴露与诗人直接关联的任何信息。"
                            "不能暴露的信息包括：具体诗句、诗句中的意向、诗人名称、诗人所处的时代、诗人与其他人的关系、诗人的字或别名等",
                            temperature=1.5)
            )

            self.save_feature = ai.DataProcessor(save_datapacket_output("WorkFlow-诗歌特征"))

            self.guess = ai.SequentialBlock(
                fct("我会给你一些诗歌内容的总结，假如这些诗歌由我创作，我在你眼里是个什么样的人？请以'我'为主语进行回答", temperature=1.6),
                ai.DataProcessor(save_datapacket_output("WorkFlow-诗人画像")),
                fct("我是一个诗人，接下来我会给你一些别人眼中我的样子，请你猜猜我是谁，并简要说明原因", temperature=1.6),
                ai.DataProcessor(save_datapacket_output("WorkFlow-诗人猜测"))
            )

        def forward(self, data):
            data = self.search(data)
            data = self.save_search(data)
            data = self.feature(data)
            data = self.save_feature(data)
            data = ai.DataPacket(content=data.metadata['content'])
            assert data.validate(str)
            data = self.guess(data)
            return data

    output = MyWorkFlow("苏轼")
    assert isinstance(output, ai.DataPacket)


def test_AIModel_FewShot():
   fct = ai.AIModelFactory(
       base_url=BASE_URL,
       api_key=API_KEY,
       model=MODEL
   )

   model1 = fct("你是一位历史学专家，用户将提供一系列问题，你的回答应当简明扼要，并以`Answer:`开头",
                few_shot_messages=[
                    {"role": "user", "content": "请问秦始皇统一六国是在哪一年？"},
                    {"role": "assistant", "content": "Answer:公元前221年"},
                    {"role": "user", "content": "请问汉朝的建立者是谁？"},
                    {"role": "assistant", "content": "Answer:刘邦"},
                    {"role": "user", "content": "请问唐朝最后一任皇帝是谁"},
                    {"role": "assistant", "content": "Answer:李柷"},
                    {"role": "user", "content": "请问明朝的开国皇帝是谁？"},
                    {"role": "assistant", "content": "Answer:朱元璋"},
                ])
   model2 = fct("你是一位历史学专家，用户将提供一系列问题，你的回答应当简明扼要，并以`Answer:`开头")

   response1 = model1("请问商朝是什么时候灭亡的")
   response2 = model2("请问商朝是什么时候灭亡的")

   data = response1.content + "\n\n---\n\n" + response2.content

   save_md("AIModel-FewShot-历史询问", data)
