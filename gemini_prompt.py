import google.generativeai as genai
import os
import PIL.Image
import json
import requests
from typing import List, Dict, Any, Optional

# Base prompt for metadata
base_prompt = "메타데이터는 다음과 같습니다:"

# Function to load images
def img_gen(len: int, path: str) -> List[PIL.Image.Image]:
    """
    Load and resize images from the specified path.

    :param len: The number of images to load.
    :param path: The path to the images.
    :return: A list of resized images.
    """
    img = []
    for i in range(len):
        img_tmp = PIL.Image.open(f'{path}{i}.png')
        img.append(resize_image(img_tmp))
    return img

# Class for the chat model
class ChatModel:
    def __init__(self):
        """
        Initialize the ChatModel with a specific vision model.
        """
        self.model = genai.GenerativeModel(model_name="models/gemini-2.0-flash-exp",
                              generation_config={"temperature": 0.4},
                              system_instruction="""위성사진과 요청사항을 입력으로 제공하면 모델이 다양한 작업을 수행해야 합니다.
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
                                                    또한, 경우에 따라 메타데이터가 json형식으로 주어질 수 있습니다. 상황에 맞게 이를 활용하여 대답을 해주세요.
                                                    """
                              )

    # Start a chat session
    def start_chat(self, history: List[Dict[str, Any]]) -> genai.ChatSession:
        """
        Start a chat session with the given history.

        :param history: The chat history as a list of dictionaries.
        :return: The chat session object.
        """
        self.chat = self.model.start_chat(history=history)
        return self.chat

    # Send a message with an image
    def send_message(self, message: str, img: Any, img_path: str) -> str:
        """
        Send a message with an image to the chat model.

        :param message: The input message as a string.
        :param img: The input image, either as a path or a PIL image.
        :param img_path: The path to the input image.
        :return: The response from the model as a string.
        """
        if isinstance(img, str):
            img = PIL.Image.open(img)
        f = genai.upload_file(img_path)
        self.response = self.chat.send_message([f, message])
        return self.response.text

# Function to generate chat history
def history_gen(img: List[PIL.Image.Image], prompt: List[Dict[str, Any]], len: int, metadata: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Generate chat history from images and prompts.

    :param img: A list of images.
    :param prompt: A list of dictionaries containing questions and answers.
    :param len: The length of the prompt list.
    :param metadata: Optional metadata to include in the prompts.
    :return: The generated chat history as a list of dictionaries.
    """
    history = []
    for i in range(len):
        if(metadata is not None):
            try:
                with open(f'{path}{i}.json', 'r', encoding='utf-8') as f:
                    meta_json = json.load(f)
                    meta_json = metadata_preprocess(meta_json)
                    meta_json = json.dumps(meta_json, ensure_ascii=False)
                    
                    meta_prompt = base_prompt + meta_json
            except FileNotFoundError:
                meta_prompt = ''
            tmp = prompt[i]['question'] + meta_prompt
        else:
            tmp = prompt[i]['question']
        history.append({"role": "user", "parts": [img[i], tmp]})
        history.append({"role": "model", "parts": prompt[i]['answer']})
    return history

# Function to resize an image
def resize_image(img: PIL.Image.Image, max_size: int = 3000) -> PIL.Image.Image:
    """
    Resize an image to a maximum size while maintaining aspect ratio.

    :param img: The input image.
    :param max_size: The maximum size for the longest dimension.
    :return: The resized image.
    """
    # 현재 이미지의 크기
    width, height = img.size

    # 가로 세로 중 가장 긴 변이 max_size보다 크다면 리사이즈 수행
    if width > max_size or height > max_size:
        # 가로와 세로 중 긴 변에 맞추어 비율을 계산
        if width > height:
            ratio = max_size / width
        else:
            ratio = max_size / height

        # 비율에 맞게 크기 조정
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        # 이미지 리사이즈
        img = img.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
        
    return img

# Function to preprocess metadata
def metadata_preprocess(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess metadata to include city and country information.

    :param metadata: The input metadata as a dictionary.
    :return: The preprocessed metadata with city and country information.
    """
    with open('./env/key.json') as f:
        auth_key = json.load(f)
        opencage_key = auth_key['opencage']
    
    # Calculate the center of the coordinates
    lonlat = metadata['geometry']['coordinates'][0]
    lon, lat = 0, 0
    for i in range(4):
        lon += lonlat[i][0]
        lat += lonlat[i][1]
    lon /= 4
    lat /= 4

    # Get location information from OpenCage API
    url = "https://api.opencagedata.com/geocode/v1/json"
    params = {
        "q": f"{lat},{lon}",
        "key": opencage_key,
        "language": "ko"  # 한국어 설정
    }

    response = requests.get(url, params=params)
    data = response.json()

    city, country = None, None
    if response.status_code == 200 and data["results"]:
        components = data["results"][0]["components"]
        city = components.get("city", components.get("town", components.get("village", "알 수 없는 도시")))
        country = components.get("country", "알 수 없는 국가")
    
    metadata['city'] = city
    metadata['country'] = country
    return metadata

# Main function
if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--len', type=int, default=2)
    parser.add_argument('--path', type=str, default='./dataset/')
    parser.add_argument('--metadata', type=str, default=None)
    args = parser.parse_args()
    len = args.len
    path = args.path
    metadata = args.metadata

    # Set API key
    with open('./env/key.json') as f:
        auth_key = json.load(f)
        genai.configure(api_key=auth_key['gemini'])

    # Get images and prompt
    img = img_gen(len, path)
    with open('./dataset/prompt.json', 'r', encoding='utf-8') as f:
        prompt = json.load(f)
        try:
            if not isinstance(prompt, list):
                raise TypeError
        except TypeError:
            print("prompt should be list type")
            exit()

    # Start chat session
    chat = ChatModel()
    chat.start_chat(history=history_gen(img, prompt, len, metadata))

    # Get input message
    input_message = input("Qustion: ")
    input_img = input("Image path(if no, say q): ")
    if input_img == 'q':
        input_img = './dataset/example.png'

    # Get metadata
    try:
        with open(input_img[:-3] + 'json', 'r', encoding='utf-8') as f:
            meta_json = json.load(f)
            meta_json = metadata_preprocess(meta_json)
            meta_json = json.dumps(meta_json, ensure_ascii=False)
            meta_prompt = base_prompt + meta_json
    except FileNotFoundError:
        meta_prompt = ''
    input_message = input_message + meta_prompt
    
    input_img_path = input_img
    input_img = resize_image(PIL.Image.open(input_img))

    # Send message and print response
    response = chat.send_message(input_message, input_img, input_img_path)
    print(response)
