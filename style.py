import streamlit as st

def set_custom_style():
    st.markdown(
        """
        <style>
        body {
            background-color: #f7f8fa;
        }
        .main {
            background-color: #fff;
            border-radius: 18px;
            padding: 2rem 2.5rem 2rem 2.5rem;
            box-shadow: 0 4px 24px 0 rgba(0,0,0,0.07);
        }
        .stButton>button, .stTextInput>div>input, .stTextArea>div>textarea {
            border-radius: 10px;
            font-size: 1.1rem;
        }
        .stSidebar {
            background: #eaf0fa;
        }
        .stSelectbox>div>div>div {
            border-radius: 10px;
        }
        h1, h2, h3 {
            color: #3a4a6b;
            font-family: 'Pretendard', 'Noto Sans KR', sans-serif;
        }
        .stMarkdown {
            font-size: 1.08rem;
        }
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {
            color: #5a6bb7;
        }
        .stButton>button {
            background-color: #e3e8fc;
            color: #3a4a6b;
            border: 1px solid #bfc8e6;
            font-weight: 600;
            transition: 0.2s;
        }
        .stButton>button:hover {
            background-color: #d1e3fa;
            color: #2d3a5b;
            border: 1.5px solid #a3b3e6;
        }
        .stTextInput>div>input, .stTextArea>div>textarea {
            background: #f4f6fb;
            border: 1px solid #dbe3f5;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def show_logo_title():
    st.markdown(
        '''
        <div style="display:flex; align-items:center; gap:14px; margin-bottom: 1.5rem;">
            <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" width="48"/>
            <span style="font-size:2.1rem; font-weight:700; color:#3a4a6b; letter-spacing:-1px;">학교생활기록부 특기사항 AI 검토 시스템</span>
        </div>
        ''',
        unsafe_allow_html=True
    )