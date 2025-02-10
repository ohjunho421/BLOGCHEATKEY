import os
import tomli
import streamlit as st
import http.client
import json
import jpype
import logging
from langchain_community.chat_models import ChatPerplexity, ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import AIMessage
from konlpy.tag import Okt
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import Counter
import re
import random

logging.basicConfig(level=logging.INFO)  # 필요할 경우 DEBUG로 변경 가능
logger = logging.getLogger(__name__)

def load_secrets():
    """시크릿 키 로드 함수"""
    with open("secrets.toml", "rb") as f:
        return tomli.load(f)

SECRETS = load_secrets()
PERPLEXITY_API_KEY = SECRETS["api"]["perplexity"]
SERPER_API_KEY = SECRETS["api"]["serper"]

@dataclass
class ConversationState:
    """대화 상태를 관리하는 데이터 클래스"""
    step: str = "keyword"
    data: Dict = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}

class ContentAnalyzer:
    def __init__(self):
        """Perplexity API 및 형태소 분석기 초기화"""
        try:
            # 만약 JVM이 아직 시작되지 않았다면 수동으로 시작합니다.
            if not jpype.isJVMStarted():
                jpype.startJVM()
            self.okt = Okt()  # Okt 형태소 분석기 초기화
        except Exception as e:
            logger.error(f"❌ 형태소 분석기(Okt) 초기화 실패: {str(e)}")
            self.okt = None  # Okt가 정상적으로 로드되지 않을 경우 대비

        self.llm = ChatPerplexity(
            model="sonar-pro",
            api_key=PERPLEXITY_API_KEY,
            temperature=0.7
        )
    
    def expand_existing_subtopics(self, content: str, keyword: str, subtopics: list) -> str:
        """글자수가 부족할 경우 기존 소제목 내용을 확장하여 보완"""
        
        # 🔹 현재 글자 수 확인 (공백 제외)
        text_length = len(content.replace(" ", ""))

        if text_length >= 1500:
            return content  # ✅ 글자수가 충분하면 그대로 반환

        expansion_prompt = f"""
        다음 글을 **1500~2000자**로 확장해주세요.

        🔹 **핵심 키워드:** {keyword}
        🔹 **소제목 목록:** 
        {'\n'.join([f"- {sub}" for sub in subtopics])}

        ✅ **확장 기준:**
        1. **기존 소제목 내에서 검색자가 궁금해할 내용을 확장** (새 소제목 추가 X)
        2. **실용적인 팁, 데이터 기반 정보, 사례 연구, FAQ 등 추가**
        3. **SEO 최적화:** `{keyword}`와 관련된 핵심 개념을 유지
        4. **문장의 흐름을 유지하면서 자연스럽게 확장**
        5. **1500~2000자 내로 유지 (불필요한 반복 금지)**

        📝 **특히 추가할 내용 예시:**  
        - `{subtopics[0]}`에 대한 실용적인 사례  
        - `{subtopics[1]}`과 관련된 최신 연구 결과 또는 통계  
        - `{subtopics[2]}`에서 자주 묻는 질문과 답변  
        - `{subtopics[3]}`에 대한 전문가 조언 및 팁  

        **원문:**
        {content}
        """

        # 🔹 Perplexity API 호출하여 확장 수행
        response = self.llm.invoke(expansion_prompt)

        if isinstance(response, AIMessage):
            return response.content
        elif isinstance(response, dict) and 'content' in response:
            return response['content']
        elif isinstance(response, str):
            return response
        else:
            return str(response)


    def analyze_morphemes(self, text: str, keyword: str = None) -> dict:
        """형태소 분석 및 출현 횟수 검증"""
        if not keyword:
            return self._analyze_basic_morphemes(text)

        # 🔹 Okt가 정상적으로 초기화되지 않았을 경우 대비
        if self.okt is None:
            raise RuntimeError("❌ 형태소 분석기(Okt)가 정상적으로 초기화되지 않았습니다.")

        morphemes = self.okt.morphs(keyword)  # ✅ self.okt 사용 가능

        analysis = {
            "is_valid": True,
            "morpheme_analysis": {},
            "needs_optimization": False
        }

        counts = {keyword: text.count(keyword)}
        for morpheme in morphemes:
            counts[morpheme] = text.count(morpheme)

        for word, count in counts.items():
            is_valid = 15 <= count <= 20
            if not is_valid:
                analysis["is_valid"] = False
                analysis["needs_optimization"] = True

            analysis["morpheme_analysis"][word] = {
                "count": count,
                "is_valid": is_valid,
                "status": "적정" if is_valid else "과다" if count > 20 else "부족"
            }

        return analysis

    def generate_content(self, prompt_data: dict) -> str:
        """LLM을 사용하여 조건에 맞는 콘텐츠 생성"""
        try:
            prompt = self._generate_writing_prompt(prompt_data)
            response = self.llm.invoke(prompt)

            if isinstance(response, AIMessage):
                content = response.content
            elif isinstance(response, dict) and 'content' in response:
                content = response['content']
            elif isinstance(response, str):
                content = response
            else:
                content = str(response)

            # 🔹 디버깅 메시지를 로그로 변경 (기본적으로 출력되지 않음)
            logger.debug(f"✅ Perplexity API 응답 타입: {type(response)}")
            logger.debug(f"✅ Perplexity API 응답 내용: {content}")

            # 🔹 형태소 분석 수행
            analysis = self.analyze_morphemes(content, prompt_data["keyword"])

            # 🔹 최적화 필요 시 수행
            if not analysis["is_valid"]:
                content = self.optimize_content(content, prompt_data["keyword"], analysis)

            # 🔹 글자수 부족 시 기존 소제목 확장 수행
            content = self.expand_existing_subtopics(content, prompt_data["keyword"], prompt_data["subtitles"])

            return content

        except Exception as e:
            logger.error(f"❌ 콘텐츠 생성 오류: {str(e)}")
            return "⚠️ 콘텐츠 생성 실패"

    def search_news(self, keyword: str, subtopics: list) -> list:
        """키워드 + 소제목 기반으로 뉴스/통계 자료 검색"""
        used_links = []
        search_terms = [keyword] + subtopics  # 키워드 + 추천된 소제목 기반 검색

        for term in search_terms:
            try:
                conn = http.client.HTTPSConnection("google.serper.dev")
                payload = json.dumps({"q": term, "gl": "kr", "hl": "ko"})

                headers = {
                    'X-API-KEY': SERPER_API_KEY,  # 🔒 secrets.toml에서 불러온 API 키 사용
                    'Content-Type': 'application/json'
                }

                conn.request("POST", "/news", payload, headers)
                res = conn.getresponse()
                data = res.read()

                response_json = json.loads(data.decode("utf-8"))
                news_results = response_json.get("news", [])

                if news_results:
                    # 🔹 최대 3개 기사만 사용
                    for article in news_results[:3]:
                        used_links.append(article.get("link", "URL 없음"))

            except Exception as e:
                print(f"⚠️ 뉴스 검색 오류: {str(e)}")

        return used_links  # 🔹 최종적으로 수집된 출처 링크 리스트 반환

        
    def _generate_writing_prompt(self, data: dict) -> str:
        """상세한 조건을 포함한 프롬프트 생성"""
        keyword = data["keyword"]
        morphemes = self.okt.morphs(keyword)

        # 🔹 안전한 데이터 가져오기
        difficulties = ', '.join(data.get("difficulties", ["관련 어려움 없음"]))
        business_name = data.get("business_name", "우리 회사")
        expertise = data.get("expertise", "전문적인 지식")

        prompt = f"""
        다음 조건을 정확히 준수하여 블로그 글을 작성해주세요:

        1. 핵심 키워드 사용 조건 (가장 중요)
        - 주 키워드: {keyword}
        - 구성 형태소: {', '.join(morphemes)}
        - 필수 출현 횟수: 키워드와 각 형태소가 각각 15-20회 출현 (ctrl+f 검색 기준)
        - 자연스러운 문맥에서 사용할 것
        
        2. 글의 구조
        - 전체 길이: 2200-2500자 (공백 제외)
        - 구조: 서론(20%) - 본론(60%) - 결론(20%)

        3. 콘텐츠 요구사항
        - 소제목: {data.get('subtitles', [])}
        - 타겟 독자: {data.get('target_audience', '')}
        - 전문성: {expertise}

        4. 형태소 사용 전략
        - 주요 개념 설명 시 키워드 전체 사용
        - 반복되는 맥락에서는 개별 형태소 활용
        - 자연스러운 문장 흐름 유지
        - 불필요한 반복 없이 고르게 분포

        5. 서론
        - 페인포인트 공감 표현
        - 전문성 강조
        - 문제 인식과 해결책 제시
        - 예시) 이 글을 읽는 여러분들은 {keyword}의 중요성에 대해 잘 알고 계신가요? 
        {keyword}는 ~에 매우 중요한 역할을 하지만, 실제로 {keyword}를 제대로 이해하고 사용하거나 
        관리하는 사람들은 많지 않습니다. 
        혹시 지금 {difficulties} 같은 문제를 겪고 계신가요?
        그렇다면 {keyword}가 원인이 될 수 있습니다.
        하지만 걱정 마시고, 이 글에 5분만 집중해주세요. 
        저희 {business_name}이(가) {expertise}를 바탕으로 
        믿을 수 있는 정보를 제공해 드리겠습니다. 
        끝까지 읽어보시고 '{difficulties}' 같은 문제를 해결해보세요.

        6. 본론
        - 통계 데이터, 사례 연구, 꿀팁 등 활용으로 신뢰도를 높일 것
        - 키워드와 형태소 활용
        - 가독성 있는 문장 구성과 글의 자연스러운 흐름으로 설득력을 높일 것

        7. 결론
        - 본론의 정보들 요약 및 정리
        - 해결이 안 될 경우 {business_name}에 문의할 수 있게끔 안내
        - 검색자의 {difficulties}를 정말 해결해 줄 수 있을지에 대한 검색자의 불안감을 해소 
        위 조건들을 모두 충족하는 전문적이고 자연스러운 글을 작성해주세요.
        특히 키워드와 형태소의 출현 횟수를 정확히 지켜주세요.
        """
        
        return prompt

    from collections import Counter

    def optimize_content(self, text: str, keyword: str, analysis: dict) -> str:
        """LLM을 사용하여 콘텐츠 최적화 (형태소 & 글자수 조정)"""
        try:
            # 🔹 형태소 분석기 활용
            morphemes = self.okt.morphs(keyword)  # 키워드를 구성하는 형태소 리스트
            word_counts = Counter(self.okt.morphs(text))  # 전체 글의 형태소 출현 횟수
            
            optimization_needed = False
            adjustments = []

            # 🔹 형태소별 출현 횟수 확인 후 최적화 필요 여부 결정
            for morpheme in morphemes:
                count = word_counts[morpheme]
                if count < 17 or count > 20:  # ✅ 17~20회 유지 필요
                    optimization_needed = True
                    adjustments.append(f"- `{morpheme}`: {count}회 → 17~20회로 조정 필요")

            # 🔹 키워드 자체 ("브레이크 라이닝") 출현 횟수 확인
            keyword_count = text.count(keyword)
            if keyword_count < 17 or keyword_count > 20:
                optimization_needed = True
                adjustments.append(f"- `{keyword}`: {keyword_count}회 → 17~20회로 조정 필요")

            # 🔹 공백 제외 글자수 계산
            text_length = len(text.replace(" ", ""))
            if text_length < 1500 or text_length > 2000:
                optimization_needed = True
                adjustments.append(f"- 현재 글자수: {text_length}자 → 1500~2000자로 조정 필요")

            if not optimization_needed:
                return text  # ✅ 최적화 필요 없으면 그대로 반환

            # 🔹 최적화 프롬프트 생성
            optimization_prompt = f"""
            다음 텍스트를 최적화해주세요.
            
            🔹 키워드: {keyword}
            🔹 형태소 구성: {', '.join(morphemes)}
            🔹 조정 필요 항목:
            {'\n'.join(adjustments)}
            
            ✅ 요구사항:
            1. `{keyword}`와 각 형태소(`{', '.join(morphemes)}`)가 17~20회 등장하도록 조정
            2. 키워드 & 형태소는 글에서 가장 많이 사용된 단어여야 함
            3. 동의어나 유사어가 있다면 자연스럽게 대체
            4. 동의어나 유사어가 없으면 문장에서 제거 (문맥상 자연스러울 경우)
            5. 공백 제외 글자수를 1500~2000자로 맞추기 (부족하면 추가 설명, 꿀팁, 사례, FAQ 포함)
            6. 전체 문맥이 자연스럽도록 유지
            
            원문:
            {text}
            """

            # 🔹 Perplexity API 호출하여 최적화 수행
            response = self.llm.invoke(optimization_prompt)

            if isinstance(response, AIMessage):
                return response.content
            elif isinstance(response, dict) and 'content' in response:
                return response['content']
            elif isinstance(response, str):
                return response
            else:
                return str(response)

        except Exception as e:
            print(f"❌ 콘텐츠 최적화 중 오류 발생: {str(e)}")
            return text

        
    def _format_analysis(self, analysis: dict) -> str:
        """분석 결과를 읽기 쉽게 포맷팅"""
        result = "형태소 분석 결과:\n"
        for word, info in analysis["morpheme_analysis"].items():
            result += f"- {word}: {info['count']}회 ({info['status']})\n"
        return result

    def _get_optimization_instructions(self, analysis: dict) -> str:
        """최적화 지침 생성"""
        instructions = []
        for word, info in analysis["morpheme_analysis"].items():
            if not info["is_valid"]:
                if info["count"] > 20:
                    instructions.append(
                        f"- {word}: {info['count']}회 → 15-20회로 감소 필요"
                    )
                else:
                    instructions.append(
                        f"- {word}: {info['count']}회 → 15-20회로 증가 필요"
                    )
        return "\n".join(instructions)

    def count_chars(self, text: str) -> dict:
        """글자수 분석"""
        text_without_spaces = text.replace(" ", "")
        count = len(text_without_spaces)
        return {
            "count": count,
            "is_valid": 2200 <= count <= 2500
        }

    def extract_statistics(self, text: str) -> List[Dict]:
        """통계 데이터 추출"""
        patterns = [
            r'(\d+(?:\.\d+)?%)',  # 백분율
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:명|개|원|달러|위|배|천|만|억)',  # 한글 단위
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:people|users|dollars|times|billion|million)'  # 영문 단위
        ]
        
        statistics = []
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                statistics.append({
                    "statistic": match.group(),
                    "context": context.strip(),
                    "position": match.start()
                })
        
        return statistics
    
class BlogChainSystem:
    def __init__(self):
        """시스템 초기화"""
        secrets = load_secrets()
        self.perplexity_api_key = secrets['api']['perplexity']
        self.analyzer = ContentAnalyzer()
        self.collected_references = []
        self.setup_chains()


    def setup_chains(self):
        secrets = load_secrets()
        self.perplexity_api_key = secrets['api']['perplexity']
        self.openai_api_key = secrets['api']['openai']
        
        self.llm = ChatPerplexity(
            model="sonar-pro",
            api_key=self.perplexity_api_key,
            temperature=0.7
        )
        
        # GPT-4 품질 검사용
        self.quality_llm = ChatOpenAI(
            model="gpt-4o",
            api_key=self.openai_api_key,
            temperature=0
        )
        
        # SEO 분석 체인
        seo_template = ChatPromptTemplate.from_template("""당신은 SEO 분석 전문가입니다. 다음 키워드를 분석해주세요:
        키워드: {keyword}

        다음 형식으로 분석 결과를 제공해주세요:

        1. 주요 검색 의도: 
        (2-3문장으로 이 키워드를 검색하는 사람들의 주요 의도를 설명해주세요)

        2. 검색자가 얻고자 하는 정보:
        (가장 중요한 3가지만 bullet point로 작성해주세요)
        - 
        - 
        - 

        3. 검색자가 겪고 있는 불편함이나 어려움:
        (가장 일반적인 3가지 어려움만 bullet point로 간단하게 작성해주세요)
        - 
        - 
        - 

        모든 내용은 간단명료하게 작성해주세요.""")
        self.seo_chain = seo_template | self.llm
        
        # 시장 조사 체인
        research_template = ChatPromptTemplate.from_template("""당신은 스크랩핑 전문가입니다. 다음 키워드에 대한 최신 시장 데이터와 국내외 기사를 조사해주세요:
        키워드: {keyword}

        다음 항목들을 조사해주세요:
        1. 검색하는 사람들이 찾는 정보
        2. {keyword}관련된 주요 기사나, 통계 데이터
        3. 소비자 행동 데이터
                                                             
        응답 형식:
        각 데이터에 대해 다음 정보를 포함해주세요:
        - 데이터 내용
        - 출처 URL
        """)
        self.research_chain = research_template | self.llm
        
        # 품질 검사 체인
        self.quality_chain = ChatPromptTemplate.from_template(
            """당신은 매우 엄격한 콘텐츠 품질 관리자입니다. 
            
            다음 블로그 내용을 엄격하게 분석하고 각 항목별로 정확한 수치와 함께 결과를 보고해주세요:
            
            검사 항목:
            1. 형태소 분석 [필수]
            - 형태소 총 사용 횟수
            - 9~11회 사용 기준 충족 여부
            - 미충족 시 현재 횟수와 목표 횟수 차이
            
            2. 글자 수 검증 [필수]
            - 공백 제외 총 글자 수
            - 2200-2500자 기준 충족 여부
            - 미충족 시 현재 글자 수와 목표 범위 차이
            
            3. 구조 분석
            - 서론(20%)/본론(60%)/결론(20%) 비율 검증
            - 각 섹션별 실제 비율 계산
            - 4개의 소제목 포함 여부
            
            4. 데이터 검증
            - 본론 내 정량적 데이터 최소 2개 검증
            - 모든 통계 데이터의 출처 명시 확인
            - 출처 누락된 데이터 목록
            
            5. 서론 요소
            - 페인포인트 공감 표현
            - 전문성 강조
            - 문제 인식과 해결책 제시

            6. 결론 요소
            - 서비스 자연스러운 소개
            - 전문성 기반 신뢰감
            - 구체적 다음 단계 제시
            
            분석할 블로그 내용:
            {prompt}
            
            각 항목을 Pass/Fail로 판정하고, Fail 항목에 대해 다음 형식으로 개선 지침을 제시하세요:
            1. [문제 항목]
            2. [현재 상태]
            3. [목표 상태]
            4. [개선 행동 지침]
            
            전체 항목 중 하나라도 Fail이면 부적합 판정입니다."""
        ) | self.quality_llm

    def handle_conversation(self):
        """대화 상태에 따른 핸들러 실행"""
        state = st.session_state.conversation_state
        
        if state.step == "keyword":
            self.handle_keyword_step()
        elif state.step == "seo_analysis":
            self.handle_seo_analysis_step()
        elif state.step == "subtopics":
            self.handle_subtopics_step()
        elif state.step == "business_info":
            self.handle_business_info_step()
        elif state.step == "target_audience":
            self.handle_target_audience_step()
        elif state.step == "morphemes":
            self.handle_morphemes_step()
        elif state.step == "reference":
            self.handle_reference_step()
        elif state.step == "content_creation":
            self.handle_content_creation_step()

    def search_related_articles(self, keyword: str, subtopics: List[str]) -> List[dict]:
        """키워드와 소제목을 기반으로 관련 기사 및 통계를 검색"""
        try:
            search_queries = [keyword] + subtopics  # 키워드 + 소제목으로 검색
            articles = []

            for query in search_queries:
                # 🔹 Perplexity 검색
                perplexity_prompt = f"'{query}'에 대한 최신 기사, 논문 또는 통계를 찾아서 요약해주세요."
                response = self.llm.invoke(perplexity_prompt)
                if isinstance(response, AIMessage):
                    articles.append({"source": "Perplexity", "content": response.content})
                elif isinstance(response, dict) and 'content' in response:
                    articles.append({"source": "Perplexity", "content": response['content']})

                # 🔹 Serper 검색
                conn = http.client.HTTPSConnection("google.serper.dev")
                payload = json.dumps({"q": query, "gl": "kr", "hl": "ko"})
                headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
                conn.request("POST", "/news", payload, headers)
                res = conn.getresponse()
                data = res.read()

                try:
                    response_json = json.loads(data.decode("utf-8"))
                    news_results = response_json.get("news", [])
                    if news_results:
                        top_article = news_results[0]  # 가장 관련성 높은 기사 1개 선택
                        articles.append({
                            "source": "Serper",
                            "title": top_article.get("title", "제목 없음"),
                            "link": top_article.get("link", "URL 없음"),
                            "summary": top_article.get("snippet", "요약 없음")
                        })
                except Exception as e:
                    print(f"⚠️ Serper 검색 중 오류 발생: {str(e)}")

            return articles

        except Exception as e:
            print(f"❌ 기사 검색 중 오류 발생: {str(e)}")
            return []

    def handle_content_creation_step(self):
        st.markdown("### 블로그 작성")
        state = st.session_state.conversation_state
        
        try:
            # 필요한 데이터 수집
            data = {
                "keyword": state.data.get('keyword', ''),
                "subtitles": state.data.get('subtopics', 
                            state.data.get('recommended_subtopics', [])),
                "target_audience": state.data.get('target_audience', {}).get('primary', ''),
                "expertise": state.data.get('business_info', {}).get('expertise', ''),
                "pain_points": state.data.get('target_audience', {}).get('pain_points', []),
                "business_name": state.data.get('business_info', {}).get('name', '')
            }

            if not data["keyword"]:
                st.error("키워드를 찾을 수 없습니다.")
                return

            progress_messages = [
                "✨ 블로그 구조를 설계하고 있습니다...",
                "📊 수집된 데이터를 분석하고 있습니다...",
                "✍️ 전문성 있는 콘텐츠를 작성하고 있습니다..."
            ]

            with st.spinner("블로그 작성을 시작합니다..."):
                for msg in progress_messages:
                    st.write(msg)
                
                # 컨텐츠 생성
                content_result = self.analyzer.generate_content(data)
                
                # 분석 수행
                morpheme_analysis = self.analyzer.analyze_morphemes(content_result, data["keyword"])
                chars_analysis = self.analyzer.count_chars(content_result)
                
                # 최적화 필요 여부 확인
                if (not morpheme_analysis["is_valid"] or not chars_analysis["is_valid"]):
                    st.info("생성된 콘텐츠를 최적화하고 있습니다...")
                    content_result = self.analyzer.optimize_content(
                        content_result, 
                        data["keyword"], 
                        morpheme_analysis
                    )
                
                # 품질 검사
                check_result = self._quality_check(content_result)

                # 결과 표시
                st.markdown("### 최종 블로그 내용")
                st.write(content_result)
                
                with st.expander("분석 결과 보기"):
                    st.write("형태소 분석:")
                    for word, info in morpheme_analysis["morpheme_analysis"].items():
                        st.write(f"- {word}: {info['count']}회 ({info['status']})")
                    
                    st.write(f"\n글자수: {chars_analysis['count']}자")
                    st.write("\n품질 검사 결과:")
                    st.write(check_result)
                
                # 옵션 버튼
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("처음부터 다시 시도", key="restart_button"):
                        st.session_state.conversation_state = ConversationState()
                        st.rerun()
                with col2:
                    if st.button("이대로 사용하기", key="use_as_is_button"):
                        st.success("블로그 글이 저장되었습니다!")

        except Exception as e:
            st.error(f"콘텐츠 생성 중 오류 발생: {str(e)}")
            print(f"🔥 디버깅: {str(e)}")

    def insert_news_sources(self, content: str, scraped_links: list) -> str:
        """사용된 뉴스 출처를 콘텐츠 최하단에 추가"""

        if not scraped_links:
            return content  # 🔹 사용된 링크가 없으면 원본 그대로 반환

        # 🔹 글에서 실제로 언급된 출처만 포함
        used_links = [link for link in scraped_links if link in content]

        if not used_links:
            return content  # 🔹 출처가 포함되지 않았다면 추가하지 않음

        # 🔹 가독성 좋은 출처 형식 생성
        sources_text = "\n\n🔗 **출처:**\n"
        for idx, link in enumerate(used_links, start=1):
            sources_text += f"{idx}. {link}\n"

        return content + sources_text  # 🔹 원본 콘텐츠에 출처 정보 추가


    def _quality_check(self, content: str) -> str:
        """품질 검사 수행"""
        try:
            response = self.quality_chain.invoke({"prompt": content})
            return str(response.content) if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"품질 검사 중 오류 발생: {str(e)}"
        
    def handle_keyword_step(self):
        """키워드 입력 및 분석 단계"""
        st.markdown("### 블로그 주제 선정")
        keyword = st.text_input("어떤 주제로 블로그를 작성하고 싶으신가요?")
        
        if keyword:
            with st.spinner("키워드 분석 중..."):
                try:
                    analysis_result = self.seo_chain.invoke({"keyword": keyword})
                    result_text = str(analysis_result.content) if hasattr(analysis_result, 'content') else str(analysis_result)
                    
                    # 분석 결과에서 실제 내용만 추출
                    if "content='" in result_text:
                        result_text = result_text.split("content='")[1].split("additional_kwargs")[0]
                    
                    # 이스케이프된 줄바꿈 처리
                    result_text = result_text.replace('\\n', '\n').strip()
                    
                    # 주요 정보 추출
                    main_intent = ""
                    pain_points = []
                    
                    # 주요 검색 의도 추출
                    if "1. 주요 검색 의도:" in result_text:
                        intent_section = result_text.split("1. 주요 검색 의도:")[1].split("2.")[0]
                        main_intent = intent_section.strip()
                    
                    # 어려움/불편함 추출
                    if "3. 검색자가 겪고 있는 불편함이나 어려움:" in result_text:
                        difficulties = result_text.split("3. 검색자가 겪고 있는 불편함이나 어려움:")[1].split("\n")
                        pain_points = [d.strip().replace('- ', '') for d in difficulties if d.strip().startswith('-')]
                    
                    # 분석 데이터 저장
                    analysis_data = {
                        'raw_text': result_text,
                        'main_intent': main_intent,
                        'pain_points': pain_points
                    }
                    
                    st.session_state.conversation_state.data['keyword'] = keyword
                    st.session_state.conversation_state.data['keyword_analysis'] = analysis_data
                    st.session_state.conversation_state.step = "seo_analysis"
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"키워드 분석 중 오류가 발생했습니다: {str(e)}")

    def handle_seo_analysis_step(self):
        st.markdown("### 검색 트렌드 분석 결과")
        
        # 분석 결과 처리
        raw_analysis = st.session_state.conversation_state.data.get('keyword_analysis', '')
        
        try:
            if raw_analysis:
                # content 부분 추출 및 포맷팅
                analysis_text = str(raw_analysis.get('raw_text', ''))
                
                # 텍스트 포맷팅
                formatted_text = ""
                for line in analysis_text.split('\n'):
                    if line.startswith('1. 주요 검색 의도:'):
                        formatted_text += f"{line}\n\n"
                    elif line.startswith('3. 검색자가 겪고 있는 불편함이나 어려움:'):
                        formatted_text += f"{line}\n"
                        difficulties = raw_analysis.get('pain_points', [])
                        for diff in difficulties:
                            formatted_text += f"- {diff}\n"
                    elif line.strip() and line.startswith('-'):
                        formatted_text += f"{line}\n"
                    elif line.strip():
                        formatted_text += f"{line}\n\n"
                
                # 분석 결과 표시
                st.write(formatted_text)
                
                # 소제목 생성
                keyword = st.session_state.conversation_state.data['keyword']
                try:
                    # 시장 조사 실행
                    research_result = self.research_chain.invoke({"keyword": keyword})
                    st.session_state.conversation_state.data['market_research'] = str(research_result)
                    
                    # 소제목 추천
                    subtopics_prompt = f"""
                    검색 키워드 '{keyword}'에 대한 블로그 소제목 4개를 추천해주세요.

                    조건:
                    1. 모든 소제목은 반드시 '{keyword}'와 직접적으로 관련되어야 함
                    2. 소제목들은 논리적 순서로 구성
                    3. 각 소제목은 검색자의 실제 고민/궁금증을 해결할 수 있는 내용
                    4. 전체적으로 '{keyword}'에 대한 포괄적 이해를 제공할 수 있는 구성

                    형식:
                    1. [첫 번째 소제목]: 기초/개요
                    2. [두 번째 소제목]: 주요 정보/특징
                    3. [세 번째 소제목]: 실용적 팁/방법
                    4. [네 번째 소제목]: 선택/관리 방법
                    """
                    
                    subtopics_result = self.llm.invoke(subtopics_prompt)
                    subtopics_content = str(subtopics_result.content) if hasattr(subtopics_result, 'content') else str(subtopics_result)
                    
                    st.session_state.conversation_state.data['recommended_subtopics'] = subtopics_content
                    
                    st.markdown("### ✍️ 추천 소제목")
                    st.write(subtopics_content)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("이 소제목들로 진행하기", key="accept_subtopics"):
                            st.session_state.conversation_state.step = "business_info"
                            st.rerun()
                    with col2:
                        if st.button("소제목 직접 입력하기", key="custom_subtopics"):
                            st.session_state.conversation_state.step = "subtopics"
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"소제목 생성 중 오류가 발생했습니다: {str(e)}")
                    
        except Exception as e:
            st.error(f"데이터 처리 중 오류가 발생했습니다: {str(e)}")

    def handle_subtopics_step(self):
        """소제목 직접 입력 단계"""
        st.markdown("### 소제목 수정")
        st.write("추천된 소제목을 수정하거나 그대로 사용하실 수 있습니다.")
        
        # 이전 단계의 추천 소제목 가져오기
        keyword_analysis = st.session_state.conversation_state.data.get('recommended_subtopics', '')
        
        # 추천된 소제목을 리스트로 변환
        recommended_subtopics = []
        if keyword_analysis:
            lines = keyword_analysis.split('\n')
            for line in lines:
                line = line.strip()
                if line and line[0].isdigit() and '. ' in line:
                    # 번호만 제거하고 나머지 형식은 유지
                    subtitle = line.split('. ', 1)[1]
                    if subtitle:
                        recommended_subtopics.append(subtitle)
        
        # 4개의 소제목이 되도록 빈 문자열로 채움
        while len(recommended_subtopics) < 4:
            recommended_subtopics.append('')
            
        # 4개의 소제목 입력 필드 생성
        subtopics = []
        for i in range(4):
            default_value = recommended_subtopics[i] if i < len(recommended_subtopics) else ''
            subtopic = st.text_input(
                f"소제목 {i+1}",
                value=default_value,
                help="원하시는 대로 수정하실 수 있습니다."
            )
            if subtopic:
                subtopics.append(subtopic)
        
        col1, col2 = st.columns(2)
        with col1:
            if len(subtopics) == 4 and st.button("소제목 확정"):
                st.session_state.conversation_state.data['subtopics'] = subtopics
                st.session_state.conversation_state.step = "business_info"
                st.rerun()
        with col2:
            if st.button("이전 단계로"):
                st.session_state.conversation_state.step = "seo_analysis"
                st.rerun()

    def handle_business_info_step(self):
        """사업자 정보 입력 단계"""
        st.markdown("### 사업자 정보 입력")
        st.write("블로그에 포함될 사업자 정보를 입력해주세요.")
        
        business_name = st.text_input("상호명을 입력해주세요", 
                                    help="예시: 디지털마케팅연구소")
        expertise = st.text_input("전문성을 입력해주세요", 
                                help="예시: 10년 경력의 마케팅 전문가, 100개 이상의 프로젝트 수행")
        
        if business_name and expertise and st.button("다음 단계로"):
            st.session_state.conversation_state.data['business_info'] = {
                "name": business_name,
                "expertise": expertise
            }
            st.session_state.conversation_state.step = "target_audience"
            st.rerun()

    def handle_target_audience_step(self):
        """타겟 독자층 설정 단계"""
        st.markdown("### 타겟 독자층 설정")
        
        # 키워드 분석 결과에서 정보 추출
        analysis_data = st.session_state.conversation_state.data.get('keyword_analysis', {})
        raw_text = analysis_data.get('raw_text', '')
        default_target = ""
        default_pain_points = ""
        
        if raw_text:
            try:
                # 주요 검색 의도 추출
                if "주요 검색 의도:" in raw_text:
                    sections = raw_text.split("2. ")[0]
                    if "1. 주요 검색 의도:" in sections:
                        default_target = sections.split("1. 주요 검색 의도:")[1].strip()
                
                # 어려움/불편함 추출
                if "3. 검색자가 겪고 있는 불편함이나 어려움:" in raw_text:
                    difficulties_section = raw_text.split("3. 검색자가 겪고 있는 불편함이나 어려움:")[1]
                    difficulties = []
                    for line in difficulties_section.split('\n'):
                        if line.strip().startswith('- '):
                            difficulties.append(line.replace('- ', ''))
                    default_pain_points = '\n'.join(difficulties)
            
            except Exception as e:
                st.error(f"데이터 처리 중 오류가 발생했습니다: {str(e)}")
        
        primary_target = st.text_input(
            "주요 타겟 독자층을 입력해주세요",
            value=default_target,
            help="예시: 소상공인, 스타트업 대표, 마케팅 담당자"
        )
        
        pain_points = st.text_area(
            "타겟 독자층이 겪는 어려움을 입력해주세요",
            value=default_pain_points,
            help="예시: 마케팅 비용 부담, 고객 확보의 어려움"
        )
        
        st.write("추가 타겟 독자층이 있다면 입력해주세요 (선택사항)")
        additional_target = st.text_input("추가 타겟 독자층")
        
        if primary_target and pain_points and st.button("다음 단계로"):
            target_info = {
                "primary": primary_target,
                "pain_points": pain_points.split('\n'),
                "additional": additional_target if additional_target else None
            }
            st.session_state.conversation_state.data['target_audience'] = target_info
            st.session_state.conversation_state.step = "morphemes"
            st.rerun()

    def handle_morphemes_step(self):
        """핵심 형태소 설정 단계"""
        st.markdown("### 핵심 형태소 설정")
        st.write("블로그에 꼭 포함되었으면 하는 핵심 단어나 형태소가 있나요? (선택사항)")
        st.write("쉼표(,)로 구분하여 입력해주세요.")
        
        morphemes_input = st.text_input("핵심 형태소", help="예시: 마케팅,성과,전략,솔루션")
        
        if st.button("다음 단계로"):
            if morphemes_input:
                morphemes = [m.strip() for m in morphemes_input.split(",")]
                st.session_state.conversation_state.data['morphemes'] = morphemes
            st.session_state.conversation_state.step = "reference"
            st.rerun()

    def handle_reference_step(self):
        """참고 자료 분석 단계"""
        st.markdown("### 참고 자료 분석")
        st.write("참고하고 싶은 블로그나 기사의 URL이 있다면 입력해주세요. (선택사항)")
        
        reference_url = st.text_input("참고 URL (선택사항)")
        
        if reference_url:
            with st.spinner("참고 자료 분석 중..."):
                try:
                    reference_prompt = f"""
                    다음 URL의 콘텐츠를 분석하세요:
                    URL: {reference_url}
                    
                    다음 항목들을 분석해주세요:
                    1. 도입부(훅) 방식
                    2. 콘텐츠 구조
                    3. 스토리텔링 방식
                    4. 결론 전개 방식
                    5. 주요 설득 포인트
                    6. 정량적 데이터 및 출처
                    """
                    
                    analysis_result = self.llm.predict(reference_prompt)
                    st.session_state.conversation_state.data['reference_analysis'] = str(analysis_result)
                    st.session_state.conversation_state.data['reference_url'] = reference_url
                except Exception as e:
                    st.error(f"참고 자료 분석 중 오류가 발생했습니다: {str(e)}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👉 참고자료 없이 바로 작성"):
                st.session_state.conversation_state.step = "content_creation"
                st.rerun()
        with col2:
            if reference_url and st.button("👉 참고자료와 함께 작성"):
                st.session_state.conversation_state.step = "content_creation"
                st.rerun()

    def reset_conversation(self):
        """대화 상태를 초기화"""
        st.session_state.conversation_state = ConversationState()

def main():
    """메인 함수"""
    st.set_page_config(page_title="🧞AI 블로그 치트키", layout="wide")
    
    # conversation_state 초기화
    if 'conversation_state' not in st.session_state:
        st.session_state.conversation_state = ConversationState()
    
    # system 초기화
    if 'system' not in st.session_state:
        st.session_state.system = BlogChainSystem()

    st.title("🧞AI 블로그 치트키")
    
    # 사이드바에 진행 상황 표시
    with st.sidebar:
        st.markdown("### 진행 상황")
        steps = {
            "keyword": "1. 키워드 선정",
            "seo_analysis": "2. 검색 트렌드 분석",
            "subtopics": "3. 소제목 선정",
            "business_info": "4. 사업자 정보",
            "target_audience": "5. 타겟 독자",
            "morphemes": "6. 핵심 형태소",
            "reference": "7. 참고 자료",
            "content_creation": "8. 블로그 작성"
        }
        
        current_step = st.session_state.conversation_state.step
        for step, label in steps.items():
            if step == current_step:
                st.markdown(f"**→ {label}**")
            elif list(steps.keys()).index(step) < list(steps.keys()).index(current_step):
                st.markdown(f"✓ {label}")
            else:
                st.markdown(f"  {label}")
        
        if st.button("처음부터 다시 시작"):
            st.session_state.system.reset_conversation()
            st.rerun()
    
    # 메인 컨텐츠
    st.session_state.system.handle_conversation()

if __name__ == "__main__":
    main()