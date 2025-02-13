import gemini_prompt
import gpt_prompt
import os
import json
from pycocoevalcap.cider.cider import Cider
import google.generativeai as genai
import PIL.Image
import deepl
from konlpy.tag import Komoran
import openai
import base64
komoran = Komoran()

def get_rsicd(only_port):
    if(only_port):
        with open('./RSICD/txtclasses_rsicd/Port.txt') as f:
            img_list = f.readlines()
            img_list = [x.strip() for x in img_list]
    else:
        img_list = []
        for file in os.listdir('./RSICD/txtclasses_rsicd'):
            file_path = os.path.join('./RSICD/txtclasses_rsicd', file)
            with open(file_path) as f:
                tmp = [x.strip() for x in f.readlines()]
                img_list.extend(tmp)
    
    return img_list

def get_json(img_list):
    with open('./RSICD/dataset_rsicd.json', "r", encoding="utf-8") as f:
        img_sent = json.load(f)
    img_sent = img_sent['images']
    arr = []
    for i in img_sent:
        if i['filename'] in img_list:
            arr.append(i)
    return arr

def translate(string):
    with open('./env/key.json') as f:
        auth_key = json.load(f)
        translator = deepl.Translator(auth_key['deepl'])
    result = translator.translate_text(string, target_lang="KO")
    return result.text

def preprocess(sentence):
    # 문장이 리스트가 아닌, 문자열로 처리되도록 해야 한다.
    return komoran.morphs(sentence)

def test_gemini_fewshot(dataset, shot):
    res = {}
    gts = {}
    for data in dataset[:10]:
        file_name = data['filename']
        sentences = data['sentences']
        sentences = [x['raw'] for x in sentences]
        sentences = [translate(x) for x in sentences]
        gts[file_name] = sentences

        len = shot
        path = './dataset/'
        metadata = None

        with open('./env/key.json') as f:
            auth_key = json.load(f)
            genai.configure(api_key=auth_key['gemini'])

        #get image and prompt
        img = gemini_prompt.img_gen(len, path)
        with open('./dataset/prompt.json', 'r', encoding='utf-8') as f:
            prompt = json.load(f)
            try:
                if not isinstance(prompt, list):
                    raise TypeError
            except TypeError:
                print("prompt should be list type")
                exit()


        chat = gemini_prompt.ChatModel()
        chat.start_chat(history=gemini_prompt.history_gen(img, prompt, len, metadata))

        #get input message
        input_message = "이 사진에 대해 캡션을 달아줘"
        input_img = './RSICD/RSICD_images/' + file_name
        
        input_img_path = input_img
        input_img = gemini_prompt.resize_image(PIL.Image.open(input_img))

        response = chat.send_message(input_message, input_img, input_img_path) 
        print("응답: ", response)
        print("예시: ", gts[file_name][0])
        res[file_name] = [response]


    gts = {key: [preprocess(caption) for caption in value] for key, value in gts.items()}
    res = {key: [preprocess(caption) for caption in value] for key, value in res.items()}
    gts = {k: [" ".join(sent) for sent in v] for k, v in gts.items()}
    res = {k: [" ".join(sent) for sent in v] for k, v in res.items()}
    cider_scorer = Cider()
    score, x = cider_scorer.compute_score(gts, res)
    print(score)
    print(x)
    return score

def test_gpt_fewshot(dataset, shot, fine_tune=False):
    res = {}
    gts = {}
    for data in dataset[:10]:
        file_name = data['filename']
        sentences = data['sentences']
        sentences = [x['raw'] for x in sentences]
        sentences = [translate(x) for x in sentences]
        gts[file_name] = sentences

        len = shot
        path = './dataset/'
        metadata = None
        fine_tune_model = "ft:gpt-4o-2024-08-06:personal::B0KHkpXT"

        with open('./env/key.json') as f:
            auth_key = json.load(f)
            openai.api_key = auth_key['gpt']

        #get image and prompt
        img = gemini_prompt.img_gen(len, path)
        img = gpt_prompt.img_64(img)
        with open('./dataset/prompt.json', 'r', encoding='utf-8') as f:
            prompt = json.load(f)
            try:
                if not isinstance(prompt, list):
                    raise TypeError
            except TypeError:
                print("prompt should be list type")
                exit()
        
        if fine_tune:
            chat = gpt_prompt.ChatModel(model = fine_tune_model)
        else:
            chat = gpt_prompt.ChatModel()
        
        #chat.start_chat(history=gemini_prompt.history_gen(img, prompt, len, metadata))

        #get input message
        input_message = "이 사진에 대해 캡션을 달아줘"
        input_img = './RSICD/RSICD_images/' + file_name
        
        input_img_path = input_img
        input_img = gemini_prompt.resize_image(PIL.Image.open(input_img))

        tmplist = []
        tmplist.append(input_img)
        input_img = gpt_prompt.img_64(tmplist)

        response = chat.send_message(input_message, input_img, input_img_path, img, prompt, len)

        print("응답: ", response)
        print("예시: ", gts[file_name][0])
        res[file_name] = [response]


    gts = {key: [preprocess(caption) for caption in value] for key, value in gts.items()}
    res = {key: [preprocess(caption) for caption in value] for key, value in res.items()}
    gts = {k: [" ".join(sent) for sent in v] for k, v in gts.items()}
    res = {k: [" ".join(sent) for sent in v] for k, v in res.items()}
    cider_scorer = Cider()
    score, x = cider_scorer.compute_score(gts, res)
    print(score)
    print(x)
    return score

if __name__ == "__main__":
    #get model, type, and rsicd set
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gemini')
    parser.add_argument('--test_type', type=str, default='fewshot')
    parser.add_argument('--only_port', type=int, default=1)
    args = parser.parse_args()
    model = args.model
    test_type = args.test_type
    only_port = args.only_port

    img_list = get_rsicd(only_port)
    dataset = get_json(img_list)

    if model == 'gemini':
        if test_type == 'fewshot':
            model_output = test_gemini_fewshot(dataset, 4)
        if test_type == 'zeroshot':
            model_output = test_gemini_fewshot(dataset, 0)
    if model == 'gpt':
        if test_type == 'fewshot':
            model_output = test_gpt_fewshot(dataset, 4)
        if test_type == 'zeroshot':
            model_output = test_gpt_fewshot(dataset, 0)
        if test_type == 'fine-tune':
            model_output = test_gpt_fewshot(dataset, 0, fine_tune=True)
    
    print("Cider Score: ", model_output)