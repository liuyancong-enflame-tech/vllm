import time
from PIL import Image
from vllm import LLM
from vllm import SamplingParams


sample_params = SamplingParams(temperature=0, max_tokens=1024)


model_path = '/pretrained_models/glm-4v-9b'
llm = LLM(
    model=model_path,
    max_model_len=3072,
    trust_remote_code=True
)

prompt = f'[gMASK] <sop> <|user|> \n <|begin_of_image|> 描述这张图片 <|assistant|>'
image = Image.open("/yancong/TestImages/Snipaste_2024-06-21_09-52-29.png").convert('RGB')
start = time.time()
outputs = llm.generate(
    [{
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }] *2, sample_params
)
print(f'generate cost {time.time()-start}s')
for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
    print('-----------------')
    