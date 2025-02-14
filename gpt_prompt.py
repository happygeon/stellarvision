import openai
import base64
from typing import List, Dict, Any

# Class for the chat model
class ChatModel:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.4):
        """
        Initialize the ChatModel with a specific model and temperature.

        :param model: The name of the model to use.
        :param temperature: The temperature setting for the model.
        """
        self.model_name = model
        self.temperature = temperature
        self.messages = [{"role": "system", "content": """위성사진과 요청사항을 입력으로 제공하면 모델이 다양한 작업을 수행해야 합니다.
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
                                                    또한, 경우에 따라 메타데이터가 json형식으로 주어질 수 있습니다. 상황에 맞게 이를 활용하여 대답을 해주세요."""}]  

    def send_message(self, input_message: str, input_img: str, input_img_path: str, img: List[Any], prompt: List[Dict[str, Any]], len: int) -> str:
        """
        Send a message with an image to the chat model.

        :param input_message: The input message as a string.
        :param input_img: The input image in base64 format.
        :param input_img_path: The path to the input image.
        :param img: A list of images to be processed.
        :param prompt: A list of dictionaries containing questions and answers.
        :param len: The length of the prompt list.
        :return: The response from the model as a string.
        """
        for i in range(len):
            conv = prompt[i]
            image = img[i]

            ques = conv['question']
            ans = conv['answer']

            prompt_user = {}
            prompt_assi = {}

            prompt_user['role'] = 'user'
            prompt_user['content'] = []
            prompt_user['content'].append({"type": "text", "text": ques})
            prompt_user['content'].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}})

            prompt_assi['role'] = 'assistant'
            prompt_assi['content'] = ans

            self.messages.append(prompt_user)
            self.messages.append(prompt_assi)
        
        # Append the input message and image to the messages
        self.messages.append({"role": "user", "content": [
            {"type": "text", "text": input_message},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{input_img}"}}
        ]})

        # Create a chat completion with the model
        self.model = openai.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            temperature=self.temperature
        )
        return self.model.choices[0].message.content

# Function to convert images to base64
def img_64(img: List[Any]) -> List[str]:
    """
    Convert a list of images to base64 format.

    :param img: A list of images.
    :return: A list of base64 encoded strings.
    """
    from io import BytesIO
    for i in range(len(img)):
        with BytesIO() as buffer:
            img[i].save(buffer, format="PNG")
            img[i] = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img