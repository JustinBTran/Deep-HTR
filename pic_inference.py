import task
import deit
import deit_models
import torch
import fairseq
from fairseq import utils
from fairseq_cli import generate
from PIL import Image
import torchvision.transforms as transforms
from line_segment import segment


############################################
import argparse

import numpy as np
import os

# from visualization import visualize_and_plot
from seg_model import *

import ssl
ssl._create_default_https_context = ssl._create_unverified_context




def init(model_path, beam=5):
    global device
    global task_obj
    model, cfg, task_obj = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        arg_overrides={"beam": beam, "task": "text_recognition", "data": "", "fp16": False})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model[0].to(device)

    img_transform = transforms.Compose([
        transforms.Resize((384, 384), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    generator = task_obj.build_generator(
        model, cfg.generation, extra_gen_cls_kwargs={'lm_model': None, 'lm_weight': None}
    )

    bpe = task_obj.build_bpe(cfg.bpe)

    return model, cfg, task_obj, generator, bpe, img_transform, device


def preprocess(img_path, img_transform):
    im = Image.open(img_path).convert('RGB').resize((384, 384))
    im = img_transform(im).unsqueeze(0).to(device).float()

    sample = {
        'net_input': {"imgs": im},
    }

    return sample


def get_text(cfg, generator, model, sample, bpe):
    decoder_output = task_obj.inference_step(generator, model, sample, prefix_tokens=None, constraints=None)
    decoder_output = decoder_output[0][0]       #top1

    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
        hypo_tokens=decoder_output["tokens"].int().cpu(),
        src_str="",
        alignment=decoder_output["alignment"],
        align_dict=None,
        tgt_dict=model[0].decoder.dictionary,
        remove_bpe=cfg.common_eval.post_process,
        extra_symbols_to_ignore=generate.get_symbols_to_strip_from_output(generator),
    )

    detok_hypo_str = bpe.decode(hypo_str)

    return detok_hypo_str

def read_word_trocr(path, model, cfg, generator, bpe, img_transform):
    print(path)

    sample = preprocess(path, img_transform)
    text = get_text(cfg, generator, model, sample, bpe)
    # cleaned_text = clean_word(text)
    print(text)
    # print(cleaned_text)
    # return cleaned_text
    return text

def clean_word(word):
    if len(word) > 2 and word[-2:] == " .":
        word = word[:-2]
    return word


def main():
    segment('static/cvl.jpg')

    words = 'lines'
    paths = []

    # model_path = 'trocr-base-handwritten.pt'
    # model_path = "s3://utmist-ml-bucket/trocr-small-handwritten-best.pt"
    model_path = 'trocr-small-handwritten.pt'
    beam = 5
    model, cfg, task, generator, bpe, img_transform, device = init(model_path, beam)

    content = []
    for image in sorted(os.listdir(words)):
        if image == '.DS_Store':
            continue
        f = os.path.join(words, image)
        paths.append(f)
    paths.sort(key=os.path.getctime)

    for path in paths:
        content.append(read_word_trocr(path, model, cfg, generator, bpe, img_transform))

    with open("output.txt", "w") as f:
        f.write("; ".join(content))
    # print(" ".join(content))


if __name__ == '__main__':
    main()