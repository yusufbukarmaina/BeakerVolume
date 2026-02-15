from dataclasses import dataclass

@dataclass
class Config:
    HF_DATASET_ID = "yusufbukarmaina/Beakers1"

    TRAIN_FRAC = 0.70
    VAL_FRAC = 0.15
    TEST_FRAC = 0.15
    SEED = 42

    FLORENCE_MODEL = "microsoft/Florence-2-base"
    QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

    FLORENCE_OUT = "./outputs/florence2"
    QWEN_OUT = "./outputs/qwen2_5"

    BATCH_SIZE = 1
    GRAD_ACCUM = 8
    LR = 2e-4
    EPOCHS = 2
    FP16 = True

    QUESTION = "What is the liquid volume in mL? Answer with only a number followed by mL."
