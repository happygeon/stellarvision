import openai
import base64
import json
import os
import deepl

# System prompt for the model to follow when performing tasks
system_prompt = """위성사진과 요청사항을 입력으로 제공하면 모델이 다양한 작업을 수행해야 합니다.
                                                    예를 들어:

                                                    1. 캡셔닝(captioning): 사진의 내용을 간단한 문장으로 설명.
                                                    2. 식별(identify): 이미지에서 특정 객체를 인식.
                                                    3. 객체 탐지(object detection): 이미지 내 객체를 박스로 표시하고 해당 위치를 출력.
                                                    4. 분류(classify): 사진 속 객체를 분류.
                                                    이 외에도 입력 이미지에 따라 다른 작업이 요구될 수 있습니다.
                                                    이미지를 분석할 때 주변의 환경과 색상을 구분하여 생각한 뒤, 결과를 출력해주세요.
                                                    만약, 산과 숲 처럼 다양한 대답이 가능한 경우, 삼림과 같이 모두를 포함하는 하나의 확실한 대답을 제시해주세요.
                                                    이미지나 사진과 같은 단어는 지양하세요.
                                                    선착장 대신 항구, 배 대신 보트라는 단어를 사용하세요.
                                                    기존의 대화를 참고하여 비슷한 양식으로 결과를 출력하세요.
                                                    또한, 경우에 따라 메타데이터가 json형식으로 주어질 수 있습니다. 상황에 맞게 이를 활용하여 대답을 해주세요."""

# Function to translate a string to Korean using DeepL API
def translate(string: str) -> str:
    """
    Translate a given string to Korean using the DeepL API.

    Args:
        string (str): The string to be translated.

    Returns:
        str: The translated string in Korean.
    """
    with open('./env/key.json') as f:
        auth_key = json.load(f)
        translator = deepl.Translator(auth_key['deepl'])
    result = translator.translate_text(string, target_lang="KO")
    return result.text

# Function to get the list of images and other data from the dataset
def get_list(dataset: str) -> tuple[list[str], list[str]]:
    """
    Get the list of images and other data from the dataset.

    Args:
        dataset (str): The name of the dataset.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists - one with image filenames and one with other data.
    """
    img_list = []
    img_else = []
    for file in os.listdir('./RSICD/txtclasses_rsicd'):
        file_path = os.path.join('./RSICD/txtclasses_rsicd', file)
        with open(file_path) as f:
            tmp = [x.strip() for x in f.readlines()[:50]]
            ttmp = [x.strip() for x in f.readlines()[50:]]
            img_list.extend(tmp)
            img_else.extend(ttmp)
    return img_list, img_else

# Function to get JSON data for the images
def get_json(img_list: list[str]) -> list[dict]:
    """
    Get JSON data for the images.

    Args:
        img_list (list[str]): A list of image filenames.

    Returns:
        list[dict]: A list of dictionaries containing image metadata.
    """
    with open('./RSICD/dataset_rsicd.json', "r", encoding="utf-8") as f:
        img_sent = json.load(f)
    img_sent = img_sent['images']
    arr = []
    for i in range(len(img_list)):
        for j in img_sent:
            if j['filename'] == img_list[i]:
                arr.append(j)

    return arr

# Function to encode an image to base64
def encode_image(image_path: str) -> str:
    """
    Encode an image to a base64 string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to create a JSONL file from the image list and test list
def make_jsonl(img_list: list[str], test_list: list[dict]) -> None:
    """
    Create a JSONL file from the image list and test list.

    Args:
        img_list (list[str]): A list of image filenames.
        test_list (list[dict]): A list of dictionaries containing test data.
    """
    data = []
    leng = min(len(img_list), len(test_list))
    for i in range(leng):
        userdict = {}
        userdict['role'] = 'user'
        userdict_content = "이 사진에 대해 캡션을 달아줘"
        userdict_image = encode_image("./RSICD/RSICD_images/" + img_list[i])
        userdict['content'] = [{"type": "text", "text": userdict_content}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{userdict_image}"}}]

        assidict = {}
        assidict['role'] = "assistant"
        assidict['content'] = [{"type": "text", "text": translate(test_list[i]['sentences'][3]['raw'])}]
        
        systdict = {}
        systdict['role'] = "system"
        systdict['content'] = [{"type": "text", "text": system_prompt}]

        tmplist = [systdict, userdict, assidict]
        tmpdict = {}
        tmpdict['messages'] = tmplist
        print("진행상황: ", i, " / ", leng)
        data.append(tmpdict)
    with open('tunedata.jsonl', 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

# Main function to parse arguments and execute the script
if __name__ == "__main__":
    with open('./env/key.json') as f:
        auth_key = json.load(f)
        openai.api_key = auth_key['gpt']
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--dataset', type=str, default='RSICD')
    parser.add_argument('--is_skip', type=str, default='True')
    args = parser.parse_args()
    model = args.model
    dataset = args.dataset
    is_skip = args.is_skip

    if is_skip == 'False':
        img_list, img_else = get_list(dataset)
        test_list = get_json(img_list)
        make_jsonl(img_list, test_list)
    
    else:
        pass