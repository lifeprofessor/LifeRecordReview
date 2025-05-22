import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from style import set_custom_style, show_logo_title
from dotenv import load_dotenv
import ssl
import gc
import torch
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel
import numpy as np
from langchain.embeddings.base import Embeddings
from langchain.schema import Document

# SSL 인증서 검증 비활성화
ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables
load_dotenv()

# Configuration constants
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"  # 더 가벼운 한국어 모델
CHUNK_SIZE = 500  # 청크 크기 감소
CHUNK_OVERLAP = 50  # 오버랩 감소
SEARCH_K = 3
CHROMA_DB_DIR = "./chroma_db"
MODEL_CACHE_DIR = "./model_cache"

def download_and_cache_model():
    """모델을 다운로드하고 캐시합니다."""
    try:
        if not os.path.exists(MODEL_CACHE_DIR):
            os.makedirs(MODEL_CACHE_DIR)
            
        # 토크나이저와 모델 다운로드
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, cache_dir=MODEL_CACHE_DIR)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL, cache_dir=MODEL_CACHE_DIR)
        
        # 모델을 CPU로 이동
        model = model.to('cpu')
        
        return tokenizer, model
    except Exception as e:
        st.error(f"모델 다운로드 중 오류 발생: {str(e)}")
        raise

def get_embeddings(text, tokenizer, model):
    """텍스트의 임베딩을 생성합니다."""
    try:
        # 토큰화
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # 임베딩 생성
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        # 정규화
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # torch 텐서를 리스트로 변환
        return embeddings[0].tolist()
    except Exception as e:
        st.error(f"임베딩 생성 중 오류 발생: {str(e)}")
        raise

def validate_environment():
    """Validate required environment variables and configurations."""
    required_vars = ["ANTHROPIC_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Validate API key format (basic check)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key.startswith("sk-"):
        raise ValueError("Invalid ANTHROPIC_API_KEY format")

# Validate environment before proceeding
try:
    validate_environment()
except ValueError as e:
    st.error(f"Configuration Error: {str(e)}")
    st.stop()

set_custom_style()
show_logo_title()

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "selected_area" not in st.session_state:
    st.session_state.selected_area = None
if "academic_level" not in st.session_state:
    st.session_state.academic_level = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "model" not in st.session_state:
    st.session_state.model = None

class CustomEmbeddingFunction(Embeddings):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def embed_documents(self, texts):
        results = []
        for text in texts:
            emb = get_embeddings(text, self.tokenizer, self.model)
            results.append(emb)
        return results

    def embed_query(self, text):
        return get_embeddings(text, self.tokenizer, self.model)

def load_and_process_documents(directory):
    """Load and process markdown documents from the specified directory."""
    try:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not os.listdir(directory):
            raise ValueError(f"Directory is empty: {directory}")
        
        with st.spinner("문서를 로드하고 처리하는 중..."):
            # 메모리 관리를 위해 한 번에 하나의 파일씩 처리
            documents = []
            for file_path in os.listdir(directory):
                if file_path.endswith('.md'):
                    try:
                        with open(os.path.join(directory, file_path), 'r', encoding='utf-8') as f:
                            content = f.read()
                            documents.append(Document(page_content=content, metadata={"source": file_path}))
                    except Exception as e:
                        st.error(f"Failed to load {file_path}: {str(e)}")
                        continue
            
            if len(documents) == 0:
                raise ValueError(f"No markdown files found in {directory}")
            
            # 청크 분할
            text_splitter = MarkdownTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(documents)
            
            # 임베딩 함수 준비
            if st.session_state.tokenizer is None or st.session_state.model is None:
                try:
                    st.session_state.tokenizer, st.session_state.model = download_and_cache_model()
                except Exception as e:
                    raise RuntimeError(f"Failed to load model: {str(e)}")
            
            embedding_function = CustomEmbeddingFunction(st.session_state.tokenizer, st.session_state.model)
            
            # 벡터 스토어 생성
            os.makedirs(CHROMA_DB_DIR, exist_ok=True)
            
            try:
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embedding_function,
                    persist_directory=CHROMA_DB_DIR
                )

                # 메모리 정리
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                st.success("문서 처리가 완료되었습니다.")
                return vectorstore

            except Exception as e:
                raise RuntimeError(f"Failed to create vector store: {str(e)}")
            
    except Exception as e:
        st.error(f"Error during document processing: {str(e)}")
        raise

def create_chain(vectorstore):
    """Create the LangChain chain for processing queries."""
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": SEARCH_K}
        )
        
        # 프롬프트를 한 문장 요약 + 예시문장 추천 형태로 변경
        template = """
        당신은 고등학교 생기부 특기사항 작성 전문가입니다.
        아래 입력된 문장을 검토하여,
        - 한 문장으로 요약된 평가(장단점 및 개선점 포함)를 먼저 제시하고,
        - 마지막에 추천 예시문장만 출력해 주세요.

        입력 문장:
        {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        model = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",  # 사용 가능한 모델명으로 교체
            temperature=0,
            anthropic_api_key=api_key
        )
        
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
        )
        
        return chain
        
    except Exception as e:
        st.error(f"Error creating chain: {str(e)}")
        raise

def render_sidebar():
    """Render the sidebar with area and academic level selection."""
    st.sidebar.header("활동 영역 선택")
    area_map = {
        "자율/자치활동 특기사항": "self_governance_guidelines",
        "진로활동 특기사항": "career_activity_guidelines"
    }
    
    area = st.sidebar.selectbox(
        "검토할 활동 영역을 선택하세요",
        list(area_map.keys()),
        index=0 if st.session_state.selected_area is None else list(area_map.keys()).index(st.session_state.selected_area)
    )
    
    academic_level = st.sidebar.selectbox(
        "학업수준을 선택하세요",
        ["상위권", "중위권", "하위권"],
        index=0 if st.session_state.academic_level is None else ["상위권", "중위권", "하위권"].index(st.session_state.academic_level)
    )
    
    if st.sidebar.button("문서 로드"):
        if area not in area_map:
            st.error("유효한 활동 영역을 선택해주세요.")
            return False
        
        try:
            directory = f"data/{area_map[area]}"
            st.session_state.vectorstore = load_and_process_documents(directory)
            st.session_state.selected_area = area
            st.session_state.academic_level = academic_level
            return True
        except Exception as e:
            st.error(f"문서 로드 중 오류가 발생했습니다: {str(e)}")
            return False
    
    return True

def process_statement(statement):
    """Process the input statement and return the review results."""
    if not st.session_state.vectorstore:
        st.error("먼저 활동 영역을 선택하고 문서를 로드해주세요.")
        return
    
    if not statement:
        st.error("검토할 문장을 입력해주세요.")
        return
    
    with st.spinner("검토 중..."):
        try:
            chain = create_chain(st.session_state.vectorstore)
            response = chain.invoke(statement)
            # Claude 응답이 AIMessage일 경우 content만 추출
            if hasattr(response, "content"):
                result = response.content
            elif isinstance(response, dict) and "content" in response:
                result = response["content"]
            else:
                result = str(response)

            # Markdown 포맷 가공
            if "[평가]" in result and "[추천 예시문장]" in result:
                평가_문장 = result.split("[추천 예시문장]")[0].replace("[평가]", "### ✅ 평가")
                예시문장 = result.split("[추천 예시문장]")[1].strip()
                예시문장 = f"### ✏️ 추천 예시문장\n> {예시문장}"
                formatted = f"{평가_문장}\n\n{예시문장}"
            else:
                formatted = result  # fallback

            st.markdown(formatted)

            return None  # main에서 중복 출력 방지
        except Exception as e:
            st.error(f"검토 중 오류가 발생했습니다: {str(e)}")
            return None

def main():
    """Main function to run the Streamlit app."""
    if render_sidebar():
        st.header("생기부 특기사항 검토")
        
        statement = st.text_area(
            "검토할 문장을 입력하세요",
            height=200,
            placeholder="검토할 문장을 입력하세요..."
        )
        
        if st.button("검토하기"):
            if statement:
                process_statement(statement)
            else:
                st.warning("검토할 문장을 입력해주세요.")

if __name__ == "__main__":
    main() 