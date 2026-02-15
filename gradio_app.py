import re
import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration
from config import Config

cfg = Config()

VOL_RE = re.compile(r"(\d+(?:\.\d+)?)")

fl_proc = AutoProcessor.from_pretrained(cfg.FLORENCE_OUT)
fl_model = AutoModelForCausalLM.from_pretrained(cfg.FLORENCE_OUT).to("cuda").eval()

qw_proc = AutoProcessor.from_pretrained(cfg.QWEN_OUT)
qw_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg.QWEN_OUT).to("cuda").eval()

def parse(txt):
    m = VOL_RE.search(txt)
    if not m:
        return "Could not parse"
    return f"{float(m.group(1))} mL"

@torch.no_grad()
def predict(image, model_choice):
    if model_choice == "Florence-2":
        inp = fl_proc(text=cfg.QUESTION, images=image, return_tensors="pt").to("cuda")
        out = fl_model.generate(**inp, max_new_tokens=16)
        txt = fl_proc.tokenizer.decode(out[0], skip_special_tokens=True)
    else:
        inp = qw_proc(text=cfg.QUESTION, images=image, return_tensors="pt").to("cuda")
        out = qw_model.generate(**inp, max_new_tokens=16)
        txt = qw_proc.tokenizer.decode(out[0], skip_special_tokens=True)

    return parse(txt)

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil"),
        gr.Radio(["Florence-2","Qwen2.5-VL"])
    ],
    outputs="text",
    title="Beaker Volume Detection"
)

demo.launch(server_name="0.0.0.0", server_port=7860)
