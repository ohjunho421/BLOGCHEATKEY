import os
import tomli
import streamlit as st
from crewai import Agent, Task, Crew
from konlpy.tag import Okt
from dataclasses import dataclass
from typing import List, Dict, Optional

def load_secrets():
    with open("secrets.toml", "rb") as f:
        return tomli.load(f)

class ContentAnalyzer:
    def __init__(self):
        self.okt = Okt()

    def analyze_morphemes(self, text: str) -> dict:
        morphemes = self.okt.morphs(text)
        unique_morphemes = set(morphemes)
        return {
            "count": len(unique_morphemes),
            "is_valid": 15 <= len(unique_morphemes) <= 20,
            "morphemes": list(unique_morphemes)
        }

    def count_chars(self, text: str) -> dict:
        text_without_spaces = text.replace(" ", "")
        count = len(text_without_spaces)
        return {
            "count": count,
            "is_valid": 1500 <= count <= 2000
        }

@dataclass
class ConversationState:
    step: str = "keyword"
    data: Dict = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}

class BlogCrewSystem:
    def __init__(self):
        secrets = load_secrets()
        os.environ['OPENAI_API_KEY'] = secrets['api']['openai']
        os.environ['ANTHROPIC_API_KEY'] = secrets['api']['anthropic']
        os.environ['TAVILY_API_KEY'] = secrets['api']['tavily']

        self.analyzer = ContentAnalyzer()
        self.setup_agents()
        if 'conversation_state' not in st.session_state:
            st.session_state.conversation_state = ConversationState()

    def setup_agents(self):
        self.conversation_agent = Agent(
            role='Conversation Manager',
            goal='사용자와의 자연스러운 대화 진행',
            backstory='사용자의 의도를 파악하고 자연스러운 대화를 이끌어가는 매니저입니다',
            llm="gpt-4o"
        )

        self.seo_agent = Agent(
            role='SEO Analyst',
            goal='키워드 분석 및 검색 트렌드 파악',
            backstory='검색 데이터를 분석하고 효과적인 주제를 제안하는 전문가입니다',
            llm="gpt-4o"
        )

        self.tavily_agent = Agent(
            role='Research Specialist',
            goal='Tavily를 활용한 데이터 수집 및 분석',
            backstory='Tavily API를 활용해 정량적 데이터와 참고자료를 분석하는 전문가입니다',
            llm="gpt-4o"
        )

        self.writer = Agent(
            role='Content Writer',
            goal='SEO 최적화된 설득력 있는 블로그 포스트 작성',
            backstory="""전문적인 블로그 작성자입니다.
            서론에서는 독자의 어려움에 공감하고 해결책을 제시하며,
            본론에서는 4개의 소제목으로 구성된 실질적인 해결방안을 제시하고,
            결론에서는 자연스러운 서비스 연결을 합니다.""",
            allow_delegation=False,
            llm="claude-3-opus-20240229"
        )

        self.quality_checker = Agent(
            role='Quality Checker',
            goal='콘텐츠 품질 검사',
            backstory='콘텐츠 품질 관리자로서 모든 요구사항 충족을 확인합니다',
            llm="gpt-4o"
        )

    def handle_conversation(self):
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

    def handle_keyword_step(self):
        st.markdown("### 블로그 주제 선정")
        keyword = st.text_input("어떤 주제로 블로그를 작성하고 싶으신가요?")
        
        if keyword:
            with st.spinner("키워드 분석 중..."):
                keyword_analysis_task = Task(
                    description=f"""
                    다음 키워드에 대한 검색 트렌드를 분석하세요:
                    키워드: {keyword}
                    
                    다음 항목들을 분석해주세요:
                    1. 월간 검색량
                    2. 주요 검색 의도
                    3. 연관 검색어
                    4. 검색 트렌드 변화
                    """,
                    agent=self.seo_agent,
                    expected_output="키워드 분석 리포트"
                )
                
                crew = Crew(
                    agents=[self.seo_agent],
                    tasks=[keyword_analysis_task],
                    verbose=True
                )
                
                analysis_result = crew.kickoff()
                st.session_state.conversation_state.data['keyword'] = keyword
                st.session_state.conversation_state.data['keyword_analysis'] = str(analysis_result)
                st.session_state.conversation_state.step = "seo_analysis"
                st.rerun()

    def handle_seo_analysis_step(self):
        st.markdown("### 검색 트렌드 분석 결과")
        st.write(st.session_state.conversation_state.data['keyword_analysis'])
        
        with st.spinner("추천 소제목 생성 중..."):
            subtopics_task = Task(
                description=f"""
                키워드 분석 결과를 바탕으로 4개의 소제목을 추천해주세요.
                키워드: {st.session_state.conversation_state.data['keyword']}
                분석 결과: {st.session_state.conversation_state.data['keyword_analysis']}
                
                소제목 조건:
                1. 검색 의도에 부합
                2. 논리적 흐름 구성
                3. 문제 해결 중심
                4. SEO 최적화
                """,
                agent=self.seo_agent,
                expected_output="4개의 추천 소제목"
            )
            
            crew = Crew(
                agents=[self.seo_agent],
                tasks=[subtopics_task],
                verbose=True
            )
            
            subtopics_result = crew.kickoff()
            st.session_state.conversation_state.data['recommended_subtopics'] = str(subtopics_result)
            
            st.markdown("### 추천 소제목")
            st.write(subtopics_result)
            
            if st.button("이 소제목들로 진행하기"):
                st.session_state.conversation_state.step = "business_info"
                st.rerun()
            elif st.button("소제목 직접 입력하기"):
                st.session_state.conversation_state.step = "subtopics"
                st.rerun()

    def handle_subtopics_step(self):
        st.markdown("### 소제목 직접 입력")
        st.write("블로그에 포함할 4개의 소제목을 입력해주세요.")
        
        subtopics = []
        for i in range(4):
            subtopic = st.text_input(f"소제목 {i+1}")
            if subtopic:
                subtopics.append(subtopic)
        
        if len(subtopics) == 4 and st.button("소제목 확정"):
            st.session_state.conversation_state.data['subtopics'] = subtopics
            st.session_state.conversation_state.step = "business_info"
            st.rerun()

    def handle_business_info_step(self):
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
        st.markdown("### 타겟 독자층 설정")
        st.write("블로그의 주요 독자층에 대해 알려주세요.")
        
        primary_target = st.text_input("주요 타겟 독자층을 입력해주세요",
                                    help="예시: 소상공인, 스타트업 대표, 마케팅 담당자")
        pain_points = st.text_area("타겟 독자층이 겪는 어려움을 입력해주세요",
                                help="예시: 마케팅 비용 부담, 고객 확보의 어려움")
        
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
        st.markdown("### 참고 자료 분석")
        st.write("참고하고 싶은 블로그나 기사의 URL이 있다면 입력해주세요. (선택사항)")
        
        reference_url = st.text_input("참고 URL")
        
        if reference_url:
            with st.spinner("참고 자료 분석 중..."):
                reference_analysis_task = Task(
                    description=f"""
                    다음 URL의 콘텐츠를 분석하세요:
                    URL: {reference_url}
                    
                    다음 항목들을 분석해주세요:
                    1. 도입부(훅) 방식
                    2. 콘텐츠 구조
                    3. 스토리텔링 방식
                    4. 결론 전개 방식
                    5. 주요 설득 포인트
                    """,
                    agent=self.tavily_agent,
                    expected_output="참고 자료 분석 리포트"
                )
                
                crew = Crew(
                    agents=[self.tavily_agent],
                    tasks=[reference_analysis_task],
                    verbose=True
                )
                
                analysis_result = crew.kickoff()
                st.session_state.conversation_state.data['reference_analysis'] = str(analysis_result)
        
        if st.button("블로그 작성 시작"):
            if reference_url:
                st.session_state.conversation_state.data['reference_url'] = reference_url
            st.session_state.conversation_state.step = "content_creation"
            st.rerun()

    def handle_content_creation_step(self):
        st.markdown("### 블로그 작성")
        state = st.session_state.conversation_state
        
        with st.spinner("블로그 작성 중... (최대 3번 시도)"):
            max_attempts = 3
            best_result = None
            
            for attempt in range(max_attempts):
                st.write(f"시도 {attempt + 1}/{max_attempts}")
                
                # Writing task
                writing_task = Task(
                    description=self._generate_writing_prompt(),
                    agent=self.writer,
                    expected_output="완성된 블로그 포스트"
                )
                
                content_crew = Crew(
                    agents=[self.writer],
                    tasks=[writing_task],
                    verbose=True
                )
                
                try:
                    result = content_crew.kickoff()
                    content = str(result)
                    
                    # Quality check
                    quality_task = Task(
                        description=self._generate_quality_check_prompt(content),
                        agent=self.quality_checker,
                        expected_output="품질 검사 결과"
                    )
                    
                    check_crew = Crew(
                        agents=[self.quality_checker],
                        tasks=[quality_task],
                        verbose=True
                    )
                    
                    check_result = check_crew.kickoff()
                    
                    content_analysis = {
                        "morphemes": self.analyzer.analyze_morphemes(content),
                        "chars": self.analyzer.count_chars(content)
                    }
                    
                    if content_analysis["morphemes"]["is_valid"] and content_analysis["chars"]["is_valid"]:
                        best_result = {
                            "content": content,
                            "analysis": content_analysis,
                            "quality_check": str(check_result)
                        }
                        break
                    else:
                        best_result = {
                            "content": content,
                            "analysis": content_analysis,
                            "quality_check": str(check_result)
                        }
                        
                except Exception as e:
                    st.error(f"에러 발생: {str(e)}")
                    continue
            
            if best_result:
                st.markdown("### 최종 블로그 내용")
                st.write(best_result["content"])
                st.markdown("### 품질 검사 결과")
                st.write(best_result["quality_check"])
                st.markdown("### 형태소 및 글자수 분석")
                st.write(best_result["analysis"])


    def _generate_writing_prompt(self):
        state = st.session_state.conversation_state.data
        
        subtopics = state.get('subtopics', None)
        if not subtopics:
            subtopics = state.get('recommended_subtopics', "")
        
        prompt = f"""
        다음 정보를 바탕으로 한국어로 블로그를 작성하세요:
        1. 키워드: {state['keyword']}
        2. 키워드 분석: {state['keyword_analysis']}
        3. 사업자정보: {state['business_info']}
        4. 타겟 독자: {state['target_audience']}
        5. 소제목: {subtopics}
        6. 필수형태소: {state.get('morphemes', [])}
        
        반드시 다음 요구사항을 준수하여 작성하세요:
        1. 형식
        - 형태소: 15-20개의 고유한 형태소 사용
        - 글자수: 1500-2000자 (공백 제외)
        - 구조: 서론/본론/결론 명확히 구분

        2. 서론 (전체의 20%)
        - {state['target_audience']['pain_points']}에 대한 깊은 공감
        - {state['business_info']['expertise']} 전문성 강조
        - 해결책 제시에 대한 기대감 유발

        3. 본론 (전체의 60%)
        - 제시된 4개의 소제목으로 구성
        - 각 섹션마다 최소 1개의 정량적 데이터 포함
        - 문제의 근본 원인과 구체적인 해결책 제시

        4. 결론 (전체의 20%)
        - {state['business_info']['name']} 서비스 자연스럽게 소개
        - 전문성을 바탕으로 신뢰감 형성
        - 독자가 취할 수 있는 구체적인 다음 단계 제시
        """
        
        if 'reference_analysis' in state:
            prompt += f"\n\n참고자료 분석 결과를 반영하세요:\n{state['reference_analysis']}"
        
        return prompt

    def _generate_quality_check_prompt(self, content: str):
        return f"""
        다음 블로그 내용이 모든 요구사항을 충족하는지 검사하세요:
        
        검사 항목:
        1. 형태소 개수 (15-20개)
        2. 글자수 (1500-2000자, 공백 제외)
        3. 필수 형태소 사용
        4. 섹션별 정량자료 포함
        5. 서론/본론/결론 구조
        6. 섹션별 비중 (서론 20%, 본론 60%, 결론 20%)
        
        블로그 내용:
        {content}
        
        각 항목별로 충족 여부를 표시하고, 미충족 항목이 있다면 구체적인 개선점을 제시하세요.
        """

    def reset_conversation(self):
        """대화 상태를 초기화합니다."""
        st.session_state.conversation_state = ConversationState()

def main():
    st.set_page_config(page_title="AI 블로그 작성 도우미", layout="wide")
    
    if 'system' not in st.session_state:
        st.session_state.system = BlogCrewSystem()
    
    st.title("AI 블로그 작성 도우미")
    
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
            elif steps[step] < steps[current_step]:
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