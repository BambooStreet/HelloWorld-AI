from flask import Flask, request
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

app = Flask(__name__)

#twillio 설정
account_sid = ''
auth_token = ''
client = Client(account_sid, auth_token)

#음성 입력 받기
@app.route("/voice", methods=['GET', 'POST'])
def voice():
    try :
        response = VoiceResponse()
        
        #Gather verb를 사용하여 음성 입력을 요청
        #input = 'speech'로 음성 입력 받는다.
        #language = 'ko-KR로 설정해 한국어 음성 인식 설정
        #action = '/process_speech' 음성 입력 처리 라우트 지정
        #/process_speech 라우트 : twillio 인식한 음성을 처리
        #request.form.get('SpeechResult', '')로 인식된 텍스트를 받아오기
        gather = Gather(input='speech', language='ko-KR', timeout=3, action='/process_speech')
        gather.say("안녕하세요, 외국인 근로자를 위한 비서 HelloWorld입니다!! 무엇을 도와드릴까요?"
                   , language="ko-KR")
        response.append(gather)
        
        #response.say(message,language="ko-KR")
        return str(response)
    
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")

        return "An error occurred", 500

#Open ai 응답 제공
@app.route("/process_speech", methods=['POST'])
def process_speech():
    #유저 음성 받아오기(텍스트 형태)
    user_speech = request.form.get('SpeechResult', '')
    response = VoiceResponse()
    
    #유저 음성 여부에 따라
    if user_speech:
        #llm 설정
        ai_response = generate_ai_response(user_speech)
        # 여기에서 음성 입력에 따른 로직을 추가할 수 있습니다.
        response.say(ai_response, language="ko-KR")
        
    #음성 반환이 안되면
    else:
        response.say("음성 입력을 받지 못했습니다. 다시 시도해 주세요.", language="ko-KR")
        response.redirect('/voice')
    
    return str(response)

#GPT 연동
def generate_ai_response(user_input):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125",
                        temperature=0,  # 창의성 (0.0 ~ 2.0)
                        max_tokens=2048,  # 최대 토큰수
                        openai_api_key="")

    template_text = """
    외국인 근로자를 상담하는 챗봇입니다. 다음 지침에 따라주세요.

    쉬운 단어와 간단한 문장 구조를 사용하세요.
    복잡한 법률 용어나 행정 용어는 풀어서 설명해주세요.
    주요 정보는 유지하되, 불필요한 세부사항은 생략하세요.
    시간 순서대로 사건을 정리하여 이해하기 쉽게 해주세요.
    날짜는 표시하지 말아주세요.

    외국인 근로자들이 비슷한 상황에 처했을 때 참고할 수 있도록 유용한 정보를 제공해주세요.

    유저 : {text}
    챗봇 : 
    """

    # 템플릿 설정
    prompt_template = PromptTemplate.from_template(template_text)

    # 템플릿에 값을 채워서 프롬프트를 완성
    filled_prompt = prompt_template.format(text= user_input)
    print(filled_prompt)  
    
    output = llm.invoke(input = filled_prompt)
    
    return output.content
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
