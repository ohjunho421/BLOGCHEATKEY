import os
import tomli
import streamlit as st
from crewai import Agent, Task, Crew
from konlpy.tag import Okt

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

class BlogCrewSystem:
    def __init__(self):
        secrets = load_secrets()
        os.environ['OPENAI_API_KEY'] = secrets['api']['openai']
        os.environ['ANTHROPIC_API_KEY'] = secrets['api']['anthropic']
        os.environ['TAVILY_API_KEY'] = secrets['api']['tavily']

        self.analyzer = ContentAnalyzer()
        self.setup_agents()

    def setup_agents(self):
        self.writer = Agent(
            role='Writer',
            goal='SEO 최적화된 설득력 있는 블로그 포스트 작성',
            backstory="""전문적인 블로그 작성자입니다. 
            서론에서는 독자의 어려움에 공감하고 해결책을 제시하며,
            본론에서는 4개의 소제목으로 구성된 실질적인 해결방안을 제시하고,
            결론에서는 자연스러운 서비스 연결을 합니다.""",
            allow_delegation=False,
            llm="gpt-4"
        )

        self.researcher = Agent(
            role='Researcher',
            goal='정량적 데이터와 통계 검색',
            backstory='데이터 분석가로서 신뢰할 수 있는 통계자료를 제공합니다',
            allow_delegation=False,
            llm="gpt-4"
        )

        self.seo_expert = Agent(
            role='SEO Expert',
            goal='키워드 분석과 소제목 추천',
            backstory='검색엔진 최적화 전문가로서 효과적인 소제목을 제안합니다',
            allow_delegation=False,
            llm="gpt-4"
        )

        self.quality_checker = Agent(
            role='Quality Checker',
            goal='콘텐츠 품질 검사',
            backstory='콘텐츠 품질 관리자로서 모든 요구사항 충족을 확인합니다',
            allow_delegation=False,
            llm="gpt-4"
        )

        self.manager = Agent(
            role='Manager',
            goal='작성된 글의 조건 충족 여부 검사',
            backstory='매니저로서 작성된 글이 모든 조건을 충족하는지 검사합니다',
            allow_delegation=False,
            llm="gpt-4"
        )

    def get_user_input(self):
        st.title("블로그 포스트 생성기")
        
        # 필수 입력값
        keyword = st.text_input("키워드를 입력하세요 *")
        business_name = st.text_input("상호명을 입력하세요 *")
        expertise = st.text_input("전문성을 입력하세요 (예: 10년 경력 전문가) *")
        target_audience = st.text_input("타겟 독자층을 입력하세요 (예: 소상공인) *")
        pain_points = st.text_input("타겟 독자층이 겪는 어려움을 입력하세요 (예: 마케팅 비용 부담) *")
        
        # 선택 입력값
        morphemes_input = st.text_input("추천 형태소를 입력하세요 (쉼표로 구분) (선택사항)")
        reference_url = st.text_input("참고 URL을 입력하세요 (선택사항)")

        # 선택 입력값 처리
        morphemes = [m.strip() for m in morphemes_input.split(",")] if morphemes_input else []
        
        business_info = {"name": business_name, "expertise": expertise}
        target_audience_info = {"type": target_audience, "pain_points": [pain_points]}
        
        if st.button("블로그 생성"):
            if not all([keyword, business_name, expertise, target_audience, pain_points]):
                st.error("* 표시된 필수 입력값을 모두 입력해주세요.")
                return None
            return keyword, business_info, target_audience_info, morphemes, reference_url
        return None

    def create_blog(self, keyword: str, business_info: dict, target_audience: dict = None, 
                morphemes: list = None, reference_url: str = None):
        max_attempts = 3
        best_result = None

        for attempt in range(max_attempts):
            st.write(f"시도 {attempt + 1}/{max_attempts}")

            # 1. SEO 분석 및 소제목 추천
            seo_task = Task(
                description=f"""
                키워드 '{keyword}'에 대해 분석하고 4개의 소제목을 추천하세요.
                - 검색 의도 분석
                - 경쟁 강도 분석
                - 소제목 추천 (4개)
                """,
                agent=self.seo_expert,
                expected_output="키워드 분석 결과와 4개의 추천 소제목이 포함된 보고서"
            )

            # 2. 데이터 수집
            research_task = Task(
                description=f"""
                다음 키워드와 관련된 정량적 데이터를 수집하세요:
                키워드: {keyword}
                필요 데이터:
                - 시장 규모
                - 성공 사례
                - 통계 자료 (최소 2개)
                """,
                agent=self.researcher,
                expected_output="시장 규모, 성공 사례, 통계 자료가 포함된 리서치 보고서"
            )

            # 3. 블로그 작성
            writing_task = Task(
                description=f"""
                다음 정보를 바탕으로 한국어로 블로그를 작성하세요:
                1. 키워드: {keyword}
                2. 사업자정보: {business_info}
                3. 타겟: {target_audience}
                4. 필수형태소: {morphemes}
                5. 참고URL: {reference_url}

                반드시 다음 요구사항을 준수하여 작성하세요:
                1. 형식
                - 형태소: 15-20개의 고유한 형태소 사용
                - 글자수: 1500-2000자 (공백 제외)
                - 구조: 서론/본론/결론 명확히 구분

                2. 서론 (전체의 20%)
                - 독자의 어려움에 깊이 공감하는 내용
                - {business_info['expertise']} 전문성 강조
                - 해결책 제시에 대한 기대감 유발

                3. 본론 (전체의 60%)
                - 4개의 명확한 소제목으로 구성
                - 각 섹션마다 최소 1개의 정량적 데이터 포함
                - 문제의 근본 원인과 구체적인 해결책 제시

                4. 결론 (전체의 20%)
                - {business_info['name']} 서비스 자연스럽게 소개
                - 전문성을 바탕으로 신뢰감 형성
                - 독자가 취할 수 있는 구체적인 다음 단계 제시

                각 섹션의 비중을 정확히 지켜주세요.
                """,
                agent=self.writer,
                expected_output="요구사항을 모두 충족하는 완성된 블로그 포스트"
            )

            # 4. 품질 검사
            quality_task = Task(
                description="""
                작성된 블로그가 다음 조건을 충족하는지 검사:
                1. 형태소 개수 (15-20개)
                2. 글자수 (1500-2000자)
                3. 필수 형태소 사용
                4. 정량자료 포함
                5. 서론/본론/결론 구조
                """,
                agent=self.quality_checker,
                expected_output="품질 검사 결과 보고서"
            )

            # 5. 매니저 검사
            manager_task = Task(
                description="""
                작성된 블로그가 모든 조건을 충족하는지 검사하고, 
                조건을 충족하지 않으면 다시 작성하도록 지시하세요.
                """,
                agent=self.manager,
                expected_output="최종 검토 결과와 수정 지시사항"
            )

            # Crew 설정 부분 수정
            crew = Crew(
                agents=[self.writer, self.researcher, self.seo_expert, self.quality_checker, self.manager],
                tasks=[seo_task, research_task, writing_task],  # 먼저 콘텐츠 생성 관련 task만 실행
                verbose=True
            )

            try:
                result = crew.kickoff()
                content = str(result)
                
                # 생성된 콘텐츠에 대한 품질 검사
                check_crew = Crew(
                    agents=[self.quality_checker, self.manager],
                    tasks=[quality_task, manager_task],
                    verbose=True
                )
                
                check_result = check_crew.kickoff()
                
                # 검증
                content_analysis = {
                    "morphemes": self.analyzer.analyze_morphemes(content),
                    "chars": self.analyzer.count_chars(content)
                }
                
                if content_analysis["morphemes"]["is_valid"] and content_analysis["chars"]["is_valid"]:
                    # 유효한 결과를 찾으면 저장하고 종료
                    best_result = {
                        "content": content,
                        "analysis": content_analysis
                    }
                    break
                else:
                    best_result = {
                        "content": content,
                        "analysis": content_analysis
                    }
            except Exception as e:
                st.error(f"에러 발생: {str(e)}")
                continue
            
        if best_result is None:
            st.error("3번의 시도 후에도 적절한 결과를 생성하지 못했습니다.")
            return {"content": "생성 실패", "analysis": {"morphemes": {"is_valid": False}, "chars": {"is_valid": False}}}
        
        return best_result

def main():
    system = BlogCrewSystem()
    inputs = system.get_user_input()
    
    if inputs:
        keyword, business_info, target_audience, morphemes, reference_url = inputs
        
        with st.spinner('블로그 생성 중... (최대 3번 시도)'):
            result = system.create_blog(
                keyword=keyword,
                business_info=business_info,
                target_audience=target_audience,
                morphemes=morphemes,
                reference_url=reference_url
            )
            
            st.write("### 블로그 내용:")
            st.write(result["content"])
            st.write("\n### 분석 결과:")
            st.write(result["analysis"])

if __name__ == "__main__":
    main()