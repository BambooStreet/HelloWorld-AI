from flask import Flask, request
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import json
#from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai  import OpenAIEmbeddings


import logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

app = Flask(__name__)

ES_CLOUD_ID = 
ES_USER =
ES_PASSWORD = 
ES_API_KEY = 
OPENAI_KEY = 

#twillio 설정
account_sid =
auth_token = 
client = Client(account_sid, auth_token)

#load DB
def load_db():
    torch.cuda.empty_cache()
    return ElasticsearchStore(
        es_cloud_id=ES_CLOUD_ID,
        es_user=ES_USER,
        es_password=ES_PASSWORD,
        es_api_key=ES_API_KEY,
        index_name='helloworld9',
        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
    )



# 대화 기록을 저장할 딕셔너리
conversations = {}

#GPT 연동
def generate_ai_response(conversation_history,query,db):

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125",
                        temperature=0,  # 창의성 (0.0 ~ 2.0)
                        max_tokens=4096,  # 최대 토큰수
                        openai_api_key="")

    template_text = """
    당신은 유능한 AI 어시스턴트입니다. 다음 관련 문서를 참조하여, 대화에 대한 적절한 답변을 생성해주세요.\n\n관련 문서: 
    {context}
    
    대화 기록:\n{conversation_history}

    AI:
    """

    similar_docs = db.similarity_search(query, k=3)

    # 검색된 문서의 내용을 하나의 문자열로 결합
    context = " ".join([doc.page_content for doc in similar_docs])

    # 템플릿 설정
    prompt_template = PromptTemplate.from_template(template_text)

    # 템플릿에 값을 채워서 프롬프트를 완성
    filled_prompt = prompt_template.format(context = context, conversation_history= conversation_history)
    
    output = llm.invoke(input = filled_prompt)
    
    return output.content

@app.route("/voice", methods=['GET', 'POST'])
def voice():
    try :
        response = VoiceResponse()
        gather = Gather(input='speech', language='ko-KR', timeout=3, action='/process_speech')
        gather.say("안녕하세요, 외국인 근로자를 위한 상담사 헬로우 월드 입니다!! 무엇을 도와드릴까요?"
                   , language="ko-KR")
        response.append(gather)
        
        return str(response)
    
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")

        return "An error occurred", 500
    
@app.route("/request_input", methods=['POST'])
def request_input():
    response = VoiceResponse()
    response.say("추가 질문이 있으시면 말씀해 주세요.", language="ko-KR")
    gather = Gather(input='speech', language='ko-KR', timeout=3, action='/process_speech')
    response.append(gather)
    return str(response)

#Open ai 응답 제공
@app.route("/process_speech", methods=['POST'])
def process_speech():
    db = load_db()
    #??
    caller_id = request.values.get('From', '')
    #유저 음성 받아오기(텍스트 형태)
    user_speech = request.values.get('SpeechResult', '')

    response = VoiceResponse()
    
    #유저 음성 입력 되면,
    if user_speech:
        # 발신자의 대화 기록이 없으면 새로 생성
        if caller_id not in conversations:
            conversations[caller_id] = []

        #사용자 입력을 대화 기록에 추가한다.
        conversations[caller_id].append({"role":"user", "content":user_speech})

        #llm 설정
        ai_response = generate_ai_response(conversations[caller_id], user_speech ,db)

        #AI응답 대화 기록에 추가
        conversations[caller_id].append({"role":"assistant","content":ai_response})

        #GPT가 생성한 음성 반환
        response.say(ai_response, language="ko-KR")

        #다시 voice 입력 받기
        #gather = Gather(input='speech', language='ko-KR', timeout=10, action='/process_speech')
        #response.append(gather)

        # 음성 재생 후 명시적으로 새로운 입력 요청
        response.redirect('/request_input')
        
        
    #음성 반환이 안되면,
    else:
        response.say("음성 입력을 받지 못했습니다. 다시 시도해 주세요.", language="ko-KR")
        response.redirect('/voice')
    
    return str(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
    
