import os
import tomli
import streamlit as st
import http.client
import json
import logging
import re
import time
import anthropic
import requests
from langchain_community.chat_models import ChatPerplexity
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain.globals import set_verbose, get_verbose
from konlpy.tag import Okt
from dataclasses import dataclass
from typing import List, Dict, Optional
from tavily import TavilyClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_secrets():
    """시크릿 키 로드 함수"""
    with open("secrets.toml", "rb") as f:
        return tomli.load(f)

SECRETS = load_secrets()
PERPLEXITY_API_KEY = SECRETS["api"]["perplexity"]
SERPER_API_KEY = SECRETS["api"]["serper"]
TAVILY_API_KEY = SECRETS["api"]["tavily"]
ANTHROPIC_API_KEY = SECRETS["api"]["anthropic"]

@dataclass
class ConversationState:
    """대화 상태를 관리하는 데이터 클래스"""
    step: str = "keyword"
    data: Dict = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}

class KeywordAnalyzer:
    """Perplexity를 사용한 키워드 분석 클래스"""
    def __init__(self):
        self.llm = ChatPerplexity(
            model="sonar-pro",
            api_key=PERPLEXITY_API_KEY,
            temperature=0.7
        )
        
    def analyze_keyword(self, keyword: str) -> dict:
        """키워드 분석 수행"""
        prompt = ChatPromptTemplate.from_template("""
        다음 키워드를 SEO 관점에서 분석해주세요:
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
        (가장 일반적인 3가지 어려움만 bullet point로 작성해주세요)
        - 
        - 
        - 
        """)
        
        result = self.llm.invoke(prompt.format(keyword=keyword))
        return self._parse_analysis_result(result)
    
    def suggest_subtopics(self, keyword: str) -> List[str]:
        """키워드 기반 소제목 추천"""
        prompt = ChatPromptTemplate.from_template("""
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
        """)
        
        result = self.llm.invoke(prompt.format(keyword=keyword))
        return self._parse_subtopics(result)

    def _parse_analysis_result(self, result) -> dict:
        """분석 결과 파싱"""
        content = str(result.content) if hasattr(result, 'content') else str(result)
        
        # 섹션별 내용 추출
        sections = content.split('\n\n')
        main_intent = ""
        info_needed = []
        pain_points = []
        
        for section in sections:
            if '1. 주요 검색 의도:' in section:
                main_intent = section.split('주요 검색 의도:')[1].strip()
            elif '2. 검색자가 얻고자 하는 정보:' in section:
                info_lines = section.split('\n')[1:]
                info_needed = [line.strip('- ').strip() for line in info_lines if line.strip().startswith('-')]
            elif '3. 검색자가 겪고 있는 불편함이나 어려움:' in section:
                pain_lines = section.split('\n')[1:]
                pain_points = [line.strip('- ').strip() for line in pain_lines if line.strip().startswith('-')]
        
        return {
            'raw_text': content,
            'main_intent': main_intent,
            'info_needed': info_needed,
            'pain_points': pain_points
        }
    
    def _parse_subtopics(self, result) -> List[str]:
        """소제목 파싱"""
        content = str(result.content) if hasattr(result, 'content') else str(result)
        subtopics = []
        
        for line in content.split('\n'):
            if line.strip() and line[0].isdigit() and '. ' in line:
                subtitle = line.split('. ', 1)[1].strip()
                if subtitle:
                    subtopics.append(subtitle)
        
        return subtopics[:4]  # 최대 4개의 소제목만 반환

class ResearchCollector:
    """Serper와 Tavily를 사용한 연구 자료 수집 클래스"""
    def __init__(self):
        self.serper_api_key = SERPER_API_KEY
        self.tavily_api_key = TAVILY_API_KEY
    
    def _get_statistics_data(self, query: str) -> List[Dict]:
        """통계 데이터 수집"""
        try:
            # 통계 데이터 추출을 위한 검색
            response = self.llm.invoke(f"Find specific statistics, numbers and data about: {query}")
            content = str(response.content) if hasattr(response, 'content') else str(response)
            
            # 통계 데이터 추출
            statistics = self._extract_statistics_from_text(content)
            
            return statistics
        except Exception as e:
            logger.error(f"통계 데이터 수집 오류: {str(e)}")
            return []

    def collect_research(self, keyword: str, subtopics: List[str]) -> Dict:
        """키워드와 소제목 관련 연구 자료 수집"""
        all_results = {
            'news': [],
            'academic': [],
            'perplexity': [],
            'statistics': []
        }
        
        # 1. 키워드 관련 자료 수집
        search_queries = [
            f"{keyword} 통계",
            f"{keyword} 연구결과",
            f"{keyword} 최신 동향",
            f"{keyword} 시장 현황",
            f"{keyword} 트렌드"
        ]
        
        # 2. 소제목 관련 자료 수집
        for subtopic in subtopics:
            search_queries.extend([
                f"{keyword} {subtopic}",
                f"{keyword} {subtopic} 통계",
                f"{keyword} {subtopic} 연구"
            ])
        
        for query in search_queries:
            # 뉴스 검색 (최신순으로 정렬)
            news_results = self._get_news_from_serper(query)
            all_results['news'].extend(news_results)
            
            # 학술 자료 검색
            academic_results = self._get_academic_from_tavily(query)
            all_results['academic'].extend(academic_results)
            
            # Perplexity 검색
            perplexity_results = self._get_perplexity_search(query)
            all_results['perplexity'].extend(perplexity_results)
        
        # 3. 통계 데이터 추출 (모든 수집된 자료에서)
        for category in ['news', 'academic', 'perplexity']:
            for item in all_results[category]:
                statistics = self._extract_statistics_from_text(
                    item.get('title', '') + ' ' + item.get('snippet', '')
                )
                if statistics:
                    for stat in statistics:
                        stat['source_url'] = item.get('url', '')
                        stat['source_title'] = item.get('title', '')
                        # 출처와 날짜 정보 추가
                        stat['source'] = item.get('source', '')
                        stat['date'] = item.get('date', '')
                    all_results['statistics'].extend(statistics)
        
        # 4. 중복 제거 및 최신순 정렬
        for category in all_results:
            all_results[category] = self._deduplicate_results(all_results[category])
            # 날짜 정보가 있는 경우 최신순 정렬
            if category in ['news', 'statistics']:
                all_results[category].sort(
                    key=lambda x: x.get('date', ''), 
                    reverse=True
                )
        
        return all_results

    def _get_perplexity_search(self, query: str, limit: int = 5) -> List[Dict]:
        """Perplexity를 통한 검색"""
        try:
            self.llm = ChatPerplexity(
                model="sonar-pro",
                api_key=PERPLEXITY_API_KEY,
                temperature=0.7
            )
            # 통계 데이터와 관련된 검색 수행
            stats_prompt = f"Find statistics, numbers, research data, news, and articles about: {query}"
            response = self.llm.invoke(stats_prompt)
            content = str(response.content) if hasattr(response, 'content') else str(response)
            
            # URL 추출
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
            
            results = []
            for url in urls[:limit]:
                results.append({
                    'title': f"Perplexity search result for: {query}",
                    'url': url,
                    'snippet': content[:200],
                    'source': 'Perplexity'
                })
            
            return results
        except Exception as e:
            logger.error(f"Perplexity 검색 오류: {str(e)}")
            return []

    def _extract_statistics_from_text(self, text: str) -> List[Dict]:
        """텍스트에서 통계 데이터 추출"""
        statistics = []
        
        # 숫자/퍼센트 패턴
        patterns = [
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:명|개|원|달러|위|배|천|만|억|%|퍼센트)',  # 한글 단위
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:people|users|dollars|percent|%)',  # 영문 단위
            r'(\d+(?:\.\d+)?)[%％]'  # 퍼센트 기호
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                # 통계 데이터의 전후 문맥 추출 (최대 100자)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                
                statistics.append({
                    'value': match.group(0),
                    'context': context,
                    'pattern_type': 'numeric' if '%' not in match.group(0) else 'percentage'
                })
        
        return statistics

    def _get_news_from_serper(self, query: str) -> List[Dict]:
        """Serper를 통한 뉴스 검색"""
        try:
            conn = http.client.HTTPSConnection("google.serper.dev")
            payload = json.dumps({
                "q": query,
                "gl": "kr",
                "hl": "ko",
                "type": "news",
                "timerange": "y"
            })
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            
            conn.request("POST", "/news", payload, headers)
            res = conn.getresponse()
            data = json.loads(res.read().decode("utf-8"))
            
            results = []
            for item in data.get("news", [])[:3]:  # 상위 3개 결과만 사용
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'date': item.get('date', ''),
                    'source': item.get('source', '')
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Serper API 오류: {str(e)}")
            return []

    def _get_academic_from_tavily(self, query: str) -> List[Dict]:
        """Tavily를 통한 학술 자료 검색"""
        try:
            client = TavilyClient(api_key=self.tavily_api_key)
            response = client.search(
                query=f"{query} research paper statistics",
                search_depth="advanced",
                time_range="year",
                include_answer="true"
            )
            
            # Tavily 응답에서 academic_papers 필터링
            results = []
            for result in response.get('results', [])[:3]:  # 상위 3개 결과만 사용
                if any(domain in result.get('url', '').lower() for domain in 
                      ['scholar.google', 'researchgate', 'academia.edu', 'sci-hub', 
                       'pubmed', 'arxiv', 'springer', 'sciencedirect']):
                    results.append({
                        'title': result.get('title', ''),
                        'url': result.get('url', ''),
                        'snippet': result.get('content', ''),
                        'score': result.get('score', 0)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Tavily API 오류: {str(e)}")
            return []

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """검색 결과 중복 제거"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        return unique_results

class ContentGenerator:
    """Claude를 사용한 콘텐츠 생성 및 최적화 클래스"""
    def __init__(self):
        self.claude = ChatAnthropic(
            anthropic_api_key=ANTHROPIC_API_KEY,
            model="claude-3-opus-20240229",
            temperature=0.7,
            max_tokens=4096
        )
        self.okt = Okt()

    def generate_content(self, data: Dict) -> str:
        """콘텐츠 생성"""
        MAX_RETRIES = 3
        RETRY_DELAY = 2  # 초

        for attempt in range(MAX_RETRIES):
            try:
                prompt = self._create_content_prompt(data)
                response = self.claude.invoke(prompt)
                content = str(response)
                
                # 최적화 필요 여부 확인
                if self._needs_optimization(content, data['keyword']):
                    content = self.optimize_content(content, data)
                
                # 참고자료가 있을 경우에만 출처 추가
                if isinstance(data.get('research_data'), dict):
                    content = self.add_references(content, data['research_data'])
                    
                return content

            except anthropic.InternalServerError as e:
                if 'overloaded_error' in str(e):
                    if attempt < MAX_RETRIES - 1:  # 마지막 시도가 아니면
                        st.warning(f"서버가 혼잡합니다. {RETRY_DELAY}초 후 재시도합니다... ({attempt + 1}/{MAX_RETRIES})")
                        time.sleep(RETRY_DELAY)
                        continue
                logger.error(f"콘텐츠 생성 중 오류 발생: {str(e)}")
                raise e
            except Exception as e:
                logger.error(f"콘텐츠 생성 중 오류 발생: {str(e)}")
                raise e

    def optimize_content(self, content: str, data: Dict) -> str:
        """콘텐츠 최적화"""
        try:
            optimization_prompt = self._create_optimization_prompt(content, data)
            response = self.claude.invoke(optimization_prompt)
            
            # Claude의 응답 처리
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)

        except Exception as e:
            logger.error(f"콘텐츠 최적화 중 오류 발생: {str(e)}")
            raise e

    def _find_citation_in_content(self, content: str, source_info: Dict) -> bool:
        """본문에서 인용 여부 확인"""
        content_lower = content.lower()
        title = source_info.get('title', '').lower()
        snippet = source_info.get('snippet', '').lower()
        
        # 인용 패턴 확인
        citation_patterns = [
            "연구에 따르면",
            "통계에 의하면",
            "조사 결과",
            "보고서에 따르면",
            "발표한 자료에 따르면",
            "의 연구진은",
            "에 따르면",
            "에 의하면",
            "출처:",
            "자료:"
        ]
        
        # 1. 제목이나 스니펫에서 핵심 수치나 문구 추출
        numbers = re.findall(r'\d+(?:\.\d+)?%?', snippet)
        key_phrases = re.findall(r'[^\s,]+\s[^\s,]+\s[^\s,]+', snippet)
        
        # 2. 인용 패턴과 함께 수치/문구가 사용되었는지 확인
        for pattern in citation_patterns:
            for number in numbers:
                if f"{pattern} {number}" in content_lower:
                    return True
            for phrase in key_phrases:
                if f"{pattern} {phrase}" in content_lower:
                    return True
        
        # 3. 제목이나 스니펫의 핵심 내용이 본문에 포함되어 있는지 확인
        if (title and title in content_lower) or (snippet and snippet in content_lower):
            return True
        
        return False

    def add_references(self, content: str, research_data: Dict) -> str:
        """콘텐츠에 사용된 출처와 모든 관련 자료 추가"""
        used_sources = []
        all_sources = []
        
        # 모든 소스 수집 및 분류
        for source_type, items in research_data.items():
            if not isinstance(items, list):
                continue
                
            for item in items:
                if not isinstance(item, dict):
                    continue
                    
                title = item.get('title', '')
                url = item.get('url', '')
                snippet = item.get('snippet', '').lower()
                date = item.get('date', '')
                
                if not url:  # URL이 없는 경우 건너뛰기
                    continue
                
                source_info = {
                    'type': source_type,
                    'title': title,
                    'url': url,
                    'date': date,
                    'snippet': snippet
                }
                
                # 본문에서 사용된 소스 확인 - 새로운 매칭 로직 사용
                if self._find_citation_in_content(content, source_info):
                    used_sources.append(source_info)
                
                all_sources.append(source_info)
        
        # 참고자료 섹션 추가
        content += "\n\n---\n## 참고자료\n"
        
        # 본문에서 사용된 자료
        if used_sources:
            content += "\n### 📚 본문에서 인용된 자료\n"
            for source in used_sources:
                if source['date']:
                    content += f"- [{source['title']}]({source['url']}) ({source['date']})\n"
                else:
                    content += f"- [{source['title']}]({source['url']})\n"
        
        # 모든 관련 자료
        content += "\n### 🔍 추가 참고자료\n"
        
        # 뉴스 자료
        content += "\n#### 📰 뉴스 자료\n"
        news_sources = [s for s in all_sources if s['type'] == 'news']
        for source in news_sources:
            content += f"- [{source['title']}]({source['url']})"
            if source['date']:
                content += f" ({source['date']})"
            content += "\n"
        
        # 학술 자료
        content += "\n#### 📚 학술/연구 자료\n"
        academic_sources = [s for s in all_sources if s['type'] == 'academic']
        for source in academic_sources:
            content += f"- [{source['title']}]({source['url']})\n"
        
        # Perplexity 검색 결과
        if any(s for s in all_sources if s['type'] == 'perplexity'):
            content += "\n#### 🔍 추가 검색 결과\n"
            perplexity_sources = [s for s in all_sources if s['type'] == 'perplexity']
            for source in perplexity_sources:
                content += f"- [{source['title']}]({source['url']})\n"
        
        return content

    def count_chars(self, text: str) -> dict:
        """글자수 분석"""
        text_without_spaces = text.replace(" ", "")
        count = len(text_without_spaces)
        return {
            "count": count,
            "is_valid": 1700 <= count <= 2000
        }

    def analyze_morphemes(self, text: str, keyword: str = None, custom_morphemes: List[str] = None) -> dict:
        """형태소 분석 및 출현 횟수 검증"""
        if not keyword:
            return {}

        if self.okt is None:
            raise RuntimeError("형태소 분석기(Okt)가 정상적으로 초기화되지 않았습니다.")

        # 정확한 카운팅을 위한 전처리
        text = re.sub(r'<[^>]+>', '', text)  # HTML 태그 제거
        text = re.sub(r'[^\w\s가-힣]', ' ', text)  # 특수문자 처리 (한글 포함)
        
        # 정확한 단어 단위 카운팅 함수
        def count_exact_word(word, text):
            pattern = rf'\b{word}\b|\b{word}(?=[\s.,!?])|(?<=[\s.,!?]){word}\b'
            return len(re.findall(pattern, text))
        
        # 키워드와 형태소 출현 횟수 계산
        keyword_count = count_exact_word(keyword, text)
        morphemes = self.okt.morphs(keyword)
        
        # 사용자 지정 형태소 추가
        if custom_morphemes:
            morphemes.extend(custom_morphemes)
        morphemes = list(set(morphemes))  # 중복 제거

        analysis = {
            "is_valid": True,
            "morpheme_analysis": {},
            "needs_optimization": False
        }

        # 키워드 분석
        analysis["morpheme_analysis"][keyword] = {
            "count": keyword_count,
            "is_valid": 17 <= keyword_count <= 20,
            "status": "적정" if 17 <= keyword_count <= 20 else "과다" if keyword_count > 20 else "부족"
        }

        # 형태소 분석
        for morpheme in morphemes:
            count = count_exact_word(morpheme, text)
            is_valid = 17 <= count <= 20
            
            if not is_valid:
                analysis["is_valid"] = False
                analysis["needs_optimization"] = True

            analysis["morpheme_analysis"][morpheme] = {
                "count": count,
                "is_valid": is_valid,
                "status": "적정" if is_valid else "과다" if count > 20 else "부족"
            }

        return analysis

    def analyze_reference(self, reference_url: str) -> Dict:
        """참고 블로그 분석"""
        try:
            reference_prompt = f"""
            다음 URL의 콘텐츠를 분석하여 주요 특징을 추출해주세요:
            URL: {reference_url}
            
            분석 항목:
            1. 글의 구조와 흐름
            2. 핵심 키워드 사용 방식
            3. 주요 데이터와 통계
            4. 설득력 있는 논리 전개 방식
            5. 독자 공감대 형성 전략
            6. CTA(Call-to-Action) 방식
            
            각 항목별로 구체적인 예시와 함께 설명해주세요.
            """
            
            response = self.claude.invoke(reference_prompt)
            
            # Claude의 응답 처리
            if hasattr(response, 'content'):
                analysis = response.content
            else:
                analysis = str(response)
                
            return {
                'url': reference_url,
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"참고 블로그 분석 중 오류 발생: {str(e)}")
            return {
                'url': reference_url,
                'error': str(e)
            }
        

    def _create_content_prompt(self, data: Dict) -> str:
        """상세한 조건을 포함한 프롬프트 생성"""
        keyword = data["keyword"]
        morphemes = self.okt.morphs(keyword)
        
        # 안전한 데이터 가져오기
        target_audience = data.get('target_audience', {})
        business_info = data.get('business_info', {})
        research_data = data.get('research_data', {})
        reference_analysis = data.get('reference_analysis', {}).get('analysis', '')
        
        # 연구 자료 포맷팅 (최대 2개씩만 사용)
        research_text = ""
        if isinstance(research_data, dict):
            news = research_data.get('news', [])[:2]
            academic = research_data.get('academic', [])[:2]
            
            if news:
                research_text += "📰 뉴스 자료:\n"
                for item in news:
                    research_text += f"- {item.get('title', '')}: {item.get('snippet', '')}\n"
            
            if academic:
                research_text += "\n📚 학술 자료:\n"
                for item in academic:
                    research_text += f"- {item.get('title', '')}: {item.get('snippet', '')}\n"

        statistics_text = ""
        if isinstance(research_data.get('statistics'), list):
            statistics_text = "\n💡 활용 가능한 통계 자료:\n"
            for stat in research_data['statistics']:
                statistics_text += f"- {stat['context']} (출처: {stat['source_title']})\n"

        prompt = f"""
        다음 조건을 준수하여 전문성 있고 친근한 블로그 글을 작성해주세요:

        필수 활용 자료:
        {research_text}
        
        통계 자료 (반드시 1개 이상 활용):
        {statistics_text}

        1. 글의 구조와 형식
        - 전체 구조: 서론(20%) - 본론(60%) - 결론(20%)
        - 각 소제목은 ### 마크다운으로 표시
        - 소제목 구성:
        ### {data['subtopics'][0] if len(data['subtopics']) > 0 else '소제목1'}
        ### {data['subtopics'][1] if len(data['subtopics']) > 1 else '소제목2'}
        ### {data['subtopics'][2] if len(data['subtopics']) > 2 else '소제목3'}
        ### {data['subtopics'][3] if len(data['subtopics']) > 3 else '소제목4'}
        - 전체 길이: 1700-2000자 (공백 제외)

        2. [필수] 서론 작성 가이드
        반드시 다음 구조로 서론을 작성해주세요:
        1) 독자의 고민/문제 공감 (반드시 최신 통계나 연구 결과 인용)
        - 수집된 통계자료나 연구결과를 활용하여 문제의 심각성이나 중요성 강조
        - "최근 연구에 따르면..." 또는 "...의 통계에 의하면..."과 같은 방식으로 시작
        - "{keyword}에 대해 고민이 많으신가요?"
        - 타겟 독자의 구체적인 어려움 언급: {', '.join(target_audience.get('pain_points', []))}
        
        2) 전문가로서의 해결책 제시
        - "이런 문제는 {keyword}만 잘 알고있어도 해결되는 문제입니다"
        - "{business_info.get('name', '')}가 {business_info.get('expertise', '')}을 바탕으로 해결해드리겠습니다"
        
        3) 독자 관심 유도
        - "이 글에서는 구체적으로 다음과 같은 내용을 다룹니다" 후 소제목 미리보기
        - "5분만 투자하시면 {keyword}에 대한 모든 것을 알 수 있습니다"

        3. 글쓰기 스타일
        - 전문가의 지식을 쉽게 설명하듯이 편안한 톤 유지
        - 각 문단은 자연스럽게 다음 문단으로 연결
        - 스토리텔링 요소 활용
        - 실제 사례나 비유를 통해 이해하기 쉽게 설명

        4. 핵심 키워드 활용
        - 주 키워드: {keyword}
        - 형태소: {', '.join(morphemes)}
        - 각 키워드와 형태소 17-20회 자연스럽게 사용
        
        5. [필수] 참고 자료 활용
        - 각 소제목 섹션마다 최소 1개 이상의 관련 통계/연구 자료 반드시 인용
        - 인용할 때는 "~에 따르면", "~의 연구 결과", "~의 통계에 의하면" 등 명확한 표현 사용
        - 모든 통계와 수치는 출처를 구체적으로 명시 (예: "2024년 OO연구소의 조사에 따르면...")
        - 가능한 최신 자료를 우선적으로 활용
        - 통계나 수치를 인용할 때는 그 의미나 시사점도 함께 설명

        6. 본론 작성 가이드
        - 각 소제목마다 핵심 주제 한 줄 요약으로 시작
        - 이론 → 사례 → 실천 방법 순으로 구성
        - 참고 자료의 통계나 연구 결과를 자연스럽게 인용
        - 전문적 내용도 쉽게 풀어서 설명
        - 각 섹션 끝에서 다음 섹션으로 자연스러운 연결

        7. 결론 작성 가이드
        - 본론 내용 요약
        - 실천 가능한 다음 단계 제시
        - "{business_info.get('name', '')}가 도와드릴 수 있다"는 메시지
        - 독자와의 상호작용 유도

        8. 참고 블로그 분석 결과 반영:
        {reference_analysis}

        위 조건들을 바탕으로 전문성과 친근함이 조화된,
        읽기 쉽고 실용적인 블로그 글을 작성해주세요.
        특히 타겟 독자({target_audience.get('primary', '')})의 어려움을 해결하는데 초점을 맞춰주세요.
        """
        
        return prompt

    def _create_optimization_prompt(self, content: str, data: Dict) -> str:
        keyword = data['keyword']
        morphemes = self.okt.morphs(keyword)
        
        analysis = self.analyze_morphemes(content, keyword)
        current_counts = {word: info["count"] for word, info in analysis["morpheme_analysis"].items()}
        
        # 동적으로 예시 생성
        example_instructions = f"""
        1. 동의어/유의어로 대체:
        - '{keyword}' 또는 각 형태소를 자연스러운 동의어/유의어로 대체
        - 해당 분야의 전문용어와 일반적인 표현을 적절히 혼용
        
        2. 문맥상 자연스러운 생략:
        - "{keyword}가 중요합니다" → "중요합니다"
        - "{keyword}를 살펴보면" → "살펴보면"
        
        3. 지시어로 대체:
        - "{keyword}는" → "이것은"
        - "{keyword}의 경우" → "이 경우"
        - "이", "이것", "해당", "이러한" 등의 지시어 활용
        """

        return f"""
        다음 블로그 글을 최적화해주세요. 다음의 출현 횟수 제한을 반드시 지켜주세요:

        🎯 목표:
        1. 키워드 '{keyword}': 정확히 17-20회 사용
        2. 각 형태소({', '.join(morphemes)}): 정확히 17-20회 사용
        
        📊 현재 상태:
        {chr(10).join([f"- '{word}': {count}회" for word, count in current_counts.items()])}

        ✂️ 과다 사용된 단어 최적화 방법 (우선순위 순):
        {example_instructions}

        ⚠️ 중요:
        - 각 형태소와 키워드가 정확히 17-20회 범위 내에서 사용되어야 함
        - ctrl+f로 검색했을 때의 횟수를 기준으로 함
        - 전체 문맥의 자연스러움을 반드시 유지
        - 전문성과 가독성의 균형 유지
        - 동의어/유의어 사용을 우선으로 하고, 자연스러운 경우에만 생략이나 지시어 사용

        원문:
        {content}

        위 지침에 따라 과다 사용된 형태소들을 최적화하여 모든 형태소가 17-20회 범위 내에 들도록 
        자연스럽게 수정해주세요. 전문성은 유지하되 읽기 쉽게 수정해주세요.
        """

    def _needs_optimization(self, content: str, keyword: str) -> bool:
        """최적화 필요 여부 확인"""
        # 글자수 확인
        text_length = len(content.replace(" ", ""))
        if text_length < 1700 or text_length > 2000:
            return True
            
        # 키워드 출현 빈도 확인
        keyword_count = content.lower().count(keyword.lower())
        if keyword_count < 15 or keyword_count > 20:
            return True
            
        # 문장 길이 확인 (한 문장이 너무 길지 않은지)
        sentences = content.split('.')
        if any(len(sent.strip()) > 200 for sent in sentences):
            return True
            
        return False

    def check_quality(self, content: str, keyword: str) -> Dict:
        """콘텐츠 품질 검사"""
        return {
            'length': len(content.replace(" ", "")),
            'readability': self._check_readability(content),
            'structure': self._check_structure(content),
            'keyword_density': self._check_keyword_density(content, keyword)
        }
        
    def _check_readability(self, content: str) -> Dict:
        """가독성 검사"""
        sentences = content.split('.')
        return {
            'avg_sentence_length': sum(len(s.strip()) for s in sentences) / len(sentences),
            'long_sentences': sum(1 for s in sentences if len(s.strip()) > 100)
        }
        
    def _check_structure(self, content: str) -> Dict:
        """구조 검사"""
        sections = content.split('\n\n')
        return {
            'total_sections': len(sections),
            'has_intro': bool(sections[0]),
            'has_conclusion': bool(sections[-1])
        }
        
    def _check_keyword_density(self, content: str, keyword: str) -> Dict:
        """키워드 밀도 검사"""
        total_words = len(content.split())
        keyword_count = content.lower().count(keyword.lower())
        return {
            'keyword_count': keyword_count,
            'density': keyword_count / total_words if total_words > 0 else 0
        }

    def collect_research(self, keyword: str, subtopics: List[str]) -> Dict:
        """키워드와 소제목 관련 연구 자료 수집"""
        all_results = {
            'news': [],
            'academic': [],
            'perplexity': [],
            'statistics': []
        }
        
        # 1. 키워드 관련 자료 수집
        search_queries = [
            f"{keyword} 통계",
            f"{keyword} 연구결과",
            f"{keyword} 최신 동향",
            f"{keyword} 시장 현황",
            f"{keyword} 트렌드"
        ]
        
        # 2. 소제목 관련 자료 수집
        for subtopic in subtopics:
            search_queries.extend([
                f"{keyword} {subtopic}",
                f"{keyword} {subtopic} 통계",
                f"{keyword} {subtopic} 연구"
            ])
        
        for query in search_queries:
            # 뉴스 검색 (최신순으로 정렬)
            news_results = self._get_news_from_serper(query)
            all_results['news'].extend(news_results)
            
            # 학술 자료 검색
            academic_results = self._get_academic_from_tavily(query)
            all_results['academic'].extend(academic_results)
            
            # Perplexity 검색
            perplexity_results = self._get_perplexity_search(query)
            all_results['perplexity'].extend(perplexity_results)
        
        # 3. 통계 데이터 추출 (모든 수집된 자료에서)
        for category in ['news', 'academic', 'perplexity']:
            for item in all_results[category]:
                statistics = self._extract_statistics_from_text(
                    item.get('title', '') + ' ' + item.get('snippet', '')
                )
                if statistics:
                    for stat in statistics:
                        stat['source_url'] = item.get('url', '')
                        stat['source_title'] = item.get('title', '')
                        # 출처와 날짜 정보 추가
                        stat['source'] = item.get('source', '')
                        stat['date'] = item.get('date', '')
                    all_results['statistics'].extend(statistics)
        
        # 4. 중복 제거 및 최신순 정렬
        for category in all_results:
            all_results[category] = self._deduplicate_results(all_results[category])
            # 날짜 정보가 있는 경우 최신순 정렬
            if category in ['news', 'statistics']:
                all_results[category].sort(
                    key=lambda x: x.get('date', ''), 
                    reverse=True
                )
        
        return all_results

    def _extract_statistics_from_text(self, text: str) -> List[Dict]:
        """텍스트에서 통계 데이터 추출"""
        statistics = []
        
        # 숫자/퍼센트 패턴
        patterns = [
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:명|개|원|달러|위|배|천|만|억|%|퍼센트)',  # 한글 단위
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:people|users|dollars|percent|%)',  # 영문 단위
            r'(\d+(?:\.\d+)?)[%％]'  # 퍼센트 기호
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                # 통계 데이터의 전후 문맥 추출 (최대 100자)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                
                statistics.append({
                    'value': match.group(0),
                    'context': context,
                    'pattern_type': 'numeric' if '%' not in match.group(0) else 'percentage'
                })
        
        return statistics

class BlogChainSystem:
    """전체 시스템 관리 클래스"""
    def __init__(self):
        """시스템 초기화"""
        secrets = load_secrets()
        self.perplexity_api_key = secrets['api']['perplexity']
        self.serper_api_key = secrets['api']['serper']
        self.keyword_analyzer = KeywordAnalyzer()  
        self.research_collector = ResearchCollector() 
        self.content_generator = ContentGenerator()
        self.collected_references = []
        self.setup_chains()
        
    def handle_conversation(self):
        """대화 상태에 따른 핸들러 실행"""
        state = st.session_state.conversation_state
        
        if state.step == "keyword":
            self.handle_keyword_step()
        elif state.step == "seo_analysis":
            self.handle_seo_analysis_step()
        elif state.step == "subtopics":
            self.handle_subtopics_step()
        elif state.step == "target_audience":    
            self.handle_target_audience_step()
        elif state.step == "business_info":      
            self.handle_business_info_step()
        elif state.step == "morphemes":          
            self.handle_morphemes_step()
        elif state.step == "reference":          
            self.handle_reference_step()
        elif state.step == "content_creation":
            self.handle_content_creation_step()

    def handle_keyword_step(self):
        """키워드 입력 및 분석 단계"""
        st.markdown("### 블로그 주제 선정")
        keyword = st.text_input("어떤 주제로 블로그를 작성하고 싶으신가요?")
        
        if keyword:
            with st.spinner("키워드 분석 중..."):
                try:
                    # Perplexity로 키워드 분석
                    analysis_result = self.keyword_analyzer.analyze_keyword(keyword)
                    
                    # 분석 결과 저장
                    st.session_state.conversation_state.data['keyword'] = keyword
                    st.session_state.conversation_state.data['keyword_analysis'] = analysis_result
                    st.session_state.conversation_state.step = "seo_analysis"
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"키워드 분석 중 오류가 발생했습니다: {str(e)}")

    def handle_seo_analysis_step(self):
        """SEO 분석 결과 표시 및 소제목 추천"""
        st.markdown("### 검색 트렌드 분석 결과")
        
        analysis_data = st.session_state.conversation_state.data.get('keyword_analysis', {})
        if analysis_data:
            # 분석 결과 표시
            st.write("### 주요 검색 의도")
            st.write(analysis_data.get('main_intent', ''))
            
            st.write("### 검색자가 얻고자 하는 정보")
            for info in analysis_data.get('info_needed', []):
                st.write(f"- {info}")
            
            st.write("### 검색자의 주요 어려움")
            for pain in analysis_data.get('pain_points', []):
                st.write(f"- {pain}")
            
            # 소제목 추천
            keyword = st.session_state.conversation_state.data['keyword']
            recommended_subtopics = self.keyword_analyzer.suggest_subtopics(keyword)
            
            # 추천 소제목을 세션 데이터에 저장
            st.session_state.conversation_state.data['recommended_subtopics'] = recommended_subtopics
            
            st.markdown("### ✍️ 추천 소제목")
            for i, subtopic in enumerate(recommended_subtopics, 1):
                st.write(f"{i}. {subtopic}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("이 소제목들로 진행하기"):
                    st.session_state.conversation_state.data['subtopics'] = recommended_subtopics.copy()
                    st.session_state.conversation_state.step = "target_audience"
                    st.rerun()
            with col2:
                if st.button("소제목 직접 입력하기"):
                    # 소제목 직접 입력 단계로 이동할 때도 추천 소제목을 유지
                    st.session_state.conversation_state.step = "subtopics"
                    st.rerun()

    def handle_business_info_step(self):
        """사업자 정보 입력"""
        st.markdown("### 사업자 정보 입력")
        
        business_name = st.text_input("상호명")
        expertise = st.text_area("전문성/경력 사항")
        
        if business_name and expertise and st.button("다음 단계로"):
            # 사업자 정보 저장
            st.session_state.conversation_state.data['business_info'] = {
                'name': business_name,
                'expertise': expertise
            }
            
            # 연구 자료 수집 시작
            with st.spinner("관련된 기사나 통계자료를 수집하고 있습니다..."):
                try:
                    keyword = st.session_state.conversation_state.data['keyword']
                    # recommended_subtopics나 subtopics 중 존재하는 것 사용
                    subtopics = (st.session_state.conversation_state.data.get('subtopics', []) or 
                                st.session_state.conversation_state.data.get('recommended_subtopics', []))
                    
                    if subtopics:
                        research_data = self.research_collector.collect_research(keyword, subtopics)
                        st.session_state.conversation_state.data['research_data'] = research_data
                    else:
                        st.warning("소제목 정보를 찾을 수 없습니다.")
                        return
                except Exception as e:
                    st.error(f"연구 자료 수집 중 오류 발생: {str(e)}")
                    return
            
            # 다음 단계로 이동
            st.session_state.conversation_state.data['business_info'] = {
                "name": business_name,
                "expertise": expertise
            }
            st.session_state.conversation_state.step = "morphemes"  # 변경
            st.rerun()

    def handle_reference_step(self):
        """참고 블로그 분석 단계"""
        st.markdown("### 참고 블로그 분석")
        st.write("참고하고 싶은 블로그나 기사의 URL을 입력해주세요. (선택사항)")
        st.write("💡 참고하고 싶은 블로그의 글쓰기 스타일을 분석하여 반영합니다.")
        
        reference_url = st.text_input("참고 URL")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("블로그 작성하기"):
                # content_creation 단계로 직접 이동
                st.session_state.conversation_state.step = "content_creation"
                st.rerun()
                
        with col2:
            if reference_url and st.button("분석하기"):
                with st.spinner("참고 블로그를 분석중입니다..."):
                    try:
                        # 블로그 분석 수행
                        reference_analysis = self.content_generator.analyze_reference(reference_url)
                        
                        if 'error' not in reference_analysis:
                            st.success("블로그 분석이 완료되었습니다!")
                            
                            # 분석 결과 표시
                            with st.expander("✍️ 분석 결과 보기"):
                                st.write("### 글의 구조와 흐름")
                                st.write(reference_analysis['analysis'])
                            
                            # 분석 결과 저장
                            st.session_state.conversation_state.data['reference_analysis'] = reference_analysis
                            
                            # 분석 완료 후 다음 단계로 이동 버튼 표시
                            if st.button("다음 단계로"):
                                st.session_state.conversation_state.step = "content_creation"
                                st.rerun()
                        else:
                            st.error(f"블로그 분석 중 오류가 발생했습니다: {reference_analysis['error']}")
                    
                    except Exception as e:
                        st.error(f"블로그 분석 중 오류가 발생했습니다: {str(e)}")

        # 이전 단계로 돌아가기 버튼
        if st.button("이전으로"):
            st.session_state.conversation_state.step = "morphemes"
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
                if "1. 주요 검색 의도:" in raw_text:
                    sections = raw_text.split("1. 주요 검색 의도:")[1].split("2.")[0]
                    default_target = sections.strip()
                
                # 어려움/불편함 추출
                if "3. 검색자가 겪고 있는 불편함이나 어려움:" in raw_text:
                    difficulties_section = raw_text.split("3. 검색자가 겪고 있는 불편함이나 어려움:")[1]
                    difficulties = []
                    for line in difficulties_section.split('\n'):
                        if line.strip().startswith('- '):
                            difficulties.append(line.replace('- ', '').strip())
                    default_pain_points = '\n'.join(difficulties)
            
            except Exception as e:
                logger.error(f"데이터 처리 중 오류 발생: {str(e)}")
        
        primary_target = st.text_input(
            "주요 타겟 독자층",
            value=default_target,
            help="예시: 소상공인, 스타트업 대표, 마케팅 담당자"
        )
        
        pain_points = st.text_area(
            "타겟 독자층이 겪는 어려움",
            value=default_pain_points,
            help="쉼표(,)로 구분하여 입력해주세요."
        )
        
        additional_target = st.text_input(
            "추가 타겟 독자층 (선택사항)",
            help="예시: 예비 창업자, 마케팅 학습자"
        )
        
        if primary_target and pain_points:
            if st.button("다음 단계로"):
                target_info = {
                    "primary": primary_target,
                    "pain_points": pain_points.split('\n'),
                    "additional": additional_target if additional_target else None
                }
                st.session_state.conversation_state.data['target_audience'] = target_info
                st.session_state.conversation_state.step = "business_info"  # 변경
                st.rerun()

    def handle_subtopics_step(self):
        """소제목 직접 입력 단계"""
        st.markdown("### 소제목 선정")
        
        # 이전 단계에서 생성된 추천 소제목 가져오기
        recommended_subtopics = []
        if 'recommended_subtopics' in st.session_state.conversation_state.data:
            recommended_subtopics = st.session_state.conversation_state.data['recommended_subtopics']
        
        st.write("블로그의 소제목을 입력해주세요. (총 4개)")
        subtopics = []
        
        # 소제목 입력 필드 - 각 필드에 추천 소제목을 초기값으로 설정
        for i in range(4):
            # 추천 소제목이 있을 경우 해당 값을, 없을 경우 빈 문자열을 기본값으로 설정
            default_value = recommended_subtopics[i] if i < len(recommended_subtopics) else ""
            
            # 각 소제목의 도움말 텍스트
            help_text = {
                0: "기초 개념이나 정의를 다루는 소제목",
                1: "주요 특징이나 장점을 설명하는 소제목",
                2: "실제 활용 방법이나 팁을 다루는 소제목",
                3: "주의사항이나 추천사항을 다루는 소제목"
            }
            
            # key를 unique하게 설정하여 초기값이 제대로 적용되도록 함
            input_key = f"subtopic_input_{i}_{default_value}"
            
            subtopic = st.text_input(
                f"소제목 {i+1}",
                value=default_value,
                help=help_text[i],
                key=input_key
            )
            
            if subtopic:
                subtopics.append(subtopic)
            else:
                subtopics.append(default_value)  # 빈 값일 경우 기본값 사용
        
        # 버튼 레이아웃
        col1, col2 = st.columns(2)
        
        # 소제목 확정 버튼
        with col1:
            if st.button("소제목 확정", use_container_width=True):
                if len(subtopics) == 4 and all(subtopics):
                    st.session_state.conversation_state.data['subtopics'] = subtopics
                    st.session_state.conversation_state.step = "target_audience"
                    st.rerun()
                else:
                    st.warning("4개의 소제목을 모두 입력해주세요.")
        
        # 이전으로 버튼
        with col2:
            if st.button("이전으로", use_container_width=True):
                st.session_state.conversation_state.step = "seo_analysis"
                st.rerun()

    def handle_morphemes_step(self):
        """핵심 형태소 설정 단계"""
        st.markdown("### 핵심 형태소 설정")
        
        # 현재 키워드 가져오기
        keyword = st.session_state.conversation_state.data.get('keyword', '')
        
        # 기본 형태소 분석 결과 표시
        if keyword:
            default_morphemes = self.content_generator.okt.morphs(keyword)
            st.write(f"⚡ '{keyword}'의 기본 형태소:")
            for morpheme in default_morphemes:
                st.write(f"- {morpheme}")
        
        st.markdown("---")
        st.write("✍️ 블로그에 추가로 포함하고 싶은 핵심 단어나 형태소가 있다면 입력해주세요.")
        st.write("💡 각 형태소는 띄어쓰기로 구분하여 입력해주세요.")
        
        morphemes_input = st.text_input(
            "추가 형태소 입력",
            help="예시: 자동차 정비 안전 점검"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("다음 단계로"):
                if morphemes_input:
                    additional_morphemes = [m.strip() for m in morphemes_input.split() if m.strip()]
                    morphemes = list(set(default_morphemes + additional_morphemes))
                    st.session_state.conversation_state.data['morphemes'] = morphemes
                else:
                    st.session_state.conversation_state.data['morphemes'] = default_morphemes
                st.session_state.conversation_state.step = "reference"
                st.rerun()
        with col2:
            if st.button("이전으로"):
                st.session_state.conversation_state.step = "business_info"
                st.rerun()

    def setup_chains(self):
        secrets = load_secrets()
        self.perplexity_api_key = secrets['api']['perplexity']
        
        self.llm = ChatPerplexity(
            model="sonar-pro",
            api_key=self.perplexity_api_key,
            temperature=0.7
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

    def handle_content_creation_step(self):
        st.markdown("### 블로그 작성")
        state = st.session_state.conversation_state
        
        try:
            data = {
                "keyword": state.data.get('keyword', ''),
                "subtopics": state.data.get('subtopics', []) or state.data.get('recommended_subtopics', []),
                "target_audience": {
                    "primary": state.data.get('target_audience', {}).get('primary', ''),
                    "pain_points": state.data.get('target_audience', {}).get('pain_points', [])
                },
                "business_info": {
                    "name": state.data.get('business_info', {}).get('name', ''),
                    "expertise": state.data.get('business_info', {}).get('expertise', '')
                },
                "morphemes": state.data.get('morphemes', []),
                "reference_analysis": state.data.get('reference_analysis', {})
            }

            # 연구 자료 수집 (add_custom_references 대신 collect_research 직접 사용)
            research_data = self.research_collector.collect_research(
                data["keyword"], 
                data["subtopics"]
            )
            data["research_data"] = research_data

            progress_messages = [
                "✨ 오류가 발생하거나 불편하신점은 편하게 문의 부탁드립니다."
            ]

            with st.spinner("블로그 작성을 시작합니다..."):
                for msg in progress_messages:
                    st.write(msg)
                    
                content_result = self.content_generator.generate_content(data)
                
                # 분석 및 최적화 수행
                morpheme_analysis = self.content_generator.analyze_morphemes(
                    text=content_result, 
                    keyword=data["keyword"],
                    custom_morphemes=data.get('morphemes', [])
                )
                chars_analysis = self.content_generator.count_chars(content_result)
                
                if (not morpheme_analysis["is_valid"] or not chars_analysis["is_valid"]):
                    st.info("생성된 콘텐츠를 최적화하고 있습니다...")
                    content_result = self.content_generator.optimize_content(
                        content_result, 
                        data
                    )
                
                # 결과 표시
                st.markdown("### 최종 블로그 내용")
                st.write(content_result)
                
                # 참고자료 섹션 표시
                if "### 참고자료" in content_result:
                    with st.expander("📚 참고한 자료 보기"):
                        sources_section = content_result.split("### 참고자료")[1].strip()
                        st.markdown(sources_section)
        
        except Exception as e:
            st.error(f"콘텐츠 생성 중 오류 발생: {str(e)}")
            logger.error(f"디버깅: {str(e)}", exc_info=True)

    def reset_conversation(self):
        """대화 상태 초기화"""
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
            "keyword": "1️⃣ 키워드 선정",
            "seo_analysis": "2️⃣ 검색 트렌드 분석",
            "subtopics": "3️⃣ 소제목 선정",
            "target_audience": "4️⃣ 타겟 독자 설정",
            "business_info": "5️⃣ 사업자 정보",
            "morphemes": "6️⃣ 핵심 형태소",
            "reference": "7️⃣ 참고 블로그",
            "content_creation": "8️⃣ 블로그 작성"
        }
        
        current_step = st.session_state.conversation_state.step
        for step, label in steps.items():
            if step == current_step:
                st.markdown(f"→ {label}")
            elif list(steps.keys()).index(step) < list(steps.keys()).index(current_step):
                st.markdown(f"✓ {label}")
            else:
                st.markdown(f"  {label}")
        
        if st.button("처음부터 다시 시작"):
            st.session_state.system.reset_conversation()
            st.rerun()
    
    # 메인 컨텐츠 영역
    st.session_state.system.handle_conversation()

if __name__ == "__main__":
    main()  # 메인 함수 실행