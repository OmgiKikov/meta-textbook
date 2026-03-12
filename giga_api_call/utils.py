import json
import os
import requests
import uuid

from dotenv import load_dotenv
from langchain_gigachat import GigaChat


class GigaChatAPIWrapper:
    def __init__(self, generation_params={}):
        load_dotenv(override=True)

        self.secret_key = os.getenv('SECRET_KEY')
        self.scope = os.getenv('SCOPE')
        self.host = os.getenv('GIGA_HOST')
        self.model = os.getenv('GIGA_MODEL')
        self.access_token = None
        self.generation_params = generation_params
        self.chat = None

        self.authorize()
        # self.get_available_models()
        self.create_chat_instance()

    def authorize(self):
        headers = {
            'Authorization': f'Bearer {self.secret_key}',
            'RqUID': str(uuid.uuid1()),
            'Content-Type': 'application/x-www-form-urlencoded',
        }

        data = {
            'scope': f'{self.scope}',
        }

        response = requests.post(os.getenv('OAUTH_URL'), headers=headers, data=data, verify=False)
        try:
            data = response.json()
            access_token = data['access_token']
            self.access_token = access_token
            return access_token
        except Exception as e:
            print('API Error:', e)
            return

    def get_available_models(self):
        models_url = os.getenv('API_URL') + "models"
        res = requests.get(models_url, headers={"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}, verify=False).json()
        print('Available models', res)
 

    def create_chat_instance(self):
        params = {
            'temperature': self.generation_params.get('temperature'),
            'max_tokens':  self.generation_params.get('max_tokens')
        }
        self.chat = GigaChat(
            model=self.model,
            access_token=self.access_token,
            verify_ssl_certs=False,
            profanity_check=False,
            **params
        )

    @staticmethod
    def parse_manual_response(response):
        data = response['choices']
        texts = list(map(lambda d: d['message']['content'], data))
        return texts

    def query_manual(self, system_message, human_message, history=[], parse_response=True):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}',
        }

        context = [
            {
                "role": "system",
                "content": system_message,
            },
        ]
        query =  [{
                "role": "user",
                "content": human_message,
            }]

        payload = json.dumps({
            "model": self.model,
            "messages": context + history + query,
            **self.generation_params
        })

        response = requests.post(
            self.host, data=payload, headers=headers, verify=False)
        if response.status_code == 401:
            self.authorize()
            return self.query_manual(system_message, human_message)

        response = response.json()
        print(response)
        if parse_response:
            texts = GigaChatAPIWrapper.parse_manual_response(response)
            return texts
        return response
