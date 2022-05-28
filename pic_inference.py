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

def segment_old():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()

    net = WordDetectorNet()
    net.load_state_dict(torch.load('Word Detector Bkp/model/weights', map_location=args.device))
    net.eval()
    net.to(args.device)

    net = WordDetectorNet()
    net.load_state_dict(torch.load('Word Detector Bkp/model/weights', map_location=args.device))
    net.eval()
    net.to(args.device)

    loader = DataLoaderImgFile(Path('Word Detector Bkp/data/test'), net.input_size, args.device)
    res = evaluate(net, loader, max_aabbs=1000)

    for i, (img, aabbs) in enumerate(zip(res.batch_imgs, res.batch_aabbs)):
        f = loader.get_scale_factor(i)
        aabbs = [aabb.scale(1 / f, 1 / f) for aabb in aabbs]
        number = int(np.floor(np.log10(len(aabbs))))
        ymins = {}
        for j, aabb in enumerate(aabbs, 0):
            ymins[j] = aabb.ymin
        ymins = sorted(ymins, key=ymins.get)
        aabbs_new = []
        for ymin in ymins:
            aabbs_new.append(aabbs[ymin])
        start_line = 0
        for j in range(1, len(aabbs_new)):
            if aabbs_new[j].ymin > aabbs_new[j - 1].ymax or j == len(aabbs_new) - 1:
                if j == len(aabbs_new) - 1:
                    j += 1
                xmins = {}
                for k, aabb in enumerate(aabbs[start_line:j], start_line):
                    xmins[k] = aabb.xmin
                xmins = sorted(xmins, key=xmins.get)
                aabbs_temp = []
                for xmin in xmins:
                    aabbs_temp.append(aabbs[xmin])
                aabbs_new[start_line:j] = aabbs_temp
                start_line = j
        img = loader.get_original_img(i)
        visualize_and_plot(img, aabbs_new, i)


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