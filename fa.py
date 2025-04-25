import streamlit as st
import numpy as np
import os
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import torch

def load_documents(folder_path: str, file_type: str) -> list:
    documents = []
    if file_type == "csv":
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(folder_path, file_name)
                try:
                    loader = CSVLoader(file_path=file_path, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    st.error(f"Lỗi khi tải file {file_name}: {str(e)}")
    elif file_type == "pdf":
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(folder_path, file_name)
                try:
                    loader = PyPDFLoader(file_path=file_path)
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    st.error(f"Lỗi khi tải file {file_name}: {str(e)}")
    return documents

FOLDERS = {
    "Cloud": {"path": "data/Cloud", "file_type": "csv"},
    "Quy_trinh_VHKT": {"path": "data/Quy_trinh_VHKT", "file_type": "pdf"},
    "Quyet_dinh_ATTT": {"path": "data/Quyet_dinh_ATTT", "file_type": "pdf"}
}

st.title('AINoiBo')

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [
        {'role': 'assistant', 'content': 'Xin chào, tôi có thể giúp gì cho bạn? Dưới đây là một số câu hỏi thường gặp:'}
    ]
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = f"session_{np.random.randint(10000)}"
if 'selected_folder' not in st.session_state:
    st.session_state['selected_folder'] = None
if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None
if 'faq_clicked' not in st.session_state:
    st.session_state['faq_clicked'] = None  
if 'suggested_questions' not in st.session_state:
    st.session_state['suggested_questions'] = {}  

with st.sidebar:
    selected_folder = st.selectbox("Chọn thư mục dữ liệu:", [""] + list(FOLDERS.keys()), index=0)

if selected_folder and selected_folder != st.session_state['selected_folder']:
    st.session_state['selected_folder'] = selected_folder
    folder_info = FOLDERS[selected_folder]
    folder_path = folder_info["path"]
    file_type = folder_info["file_type"]

    if not os.path.exists(folder_path):
        st.error(f"Thư mục {folder_path} không tồn tại.")
        st.stop()

    try:
        documents = load_documents(folder_path, file_type)
        if not documents:
            st.error(f"Không tìm thấy tài liệu trong thư mục {selected_folder}.")
            st.stop()
    except Exception as e:
        st.error(f"Lỗi khi tải tài liệu: {str(e)}")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(documents)

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={"device": "mps" if torch.backends.mps.is_available() else "cpu"}
        )
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo embeddings: {str(e)}")
        st.stop()

    batch_size = 50
    vectorstore = InMemoryVectorStore(embedding=embeddings)
    try:
        for i in range(0, len(all_splits), batch_size):
            batch = all_splits[i:i + batch_size]
            vectorstore.add_documents(batch)
    except Exception as e:
        st.error(f"Lỗi khi tạo vectorstore: {str(e)}")
        st.stop()

    st.session_state['vectorstore'] = vectorstore
    st.session_state['retriever'] = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={"k": 10}
    )

try:
    llm = ChatOllama(
        model='llama3.2:3b',
        temperature=0.3,
        base_url='http://localhost:11434',
    )
except Exception as e:
    st.error(f"Không thể kết nối đến Ollama: {str(e)}")
    st.write("Vui lòng đảm bảo Ollama đang chạy (`ollama serve`) và mô hình `llama3.2:3b` đã được tải (`ollama pull llama3.2:3b`).")
    st.stop()

system_prompt = (
    "Bạn là AINoiBo, một mô hình ngôn ngữ lớn, được tạo ra bới Chuyên viên Vũ Phát Đạt - Phòng Kỹ thuật khai thác - Trung tâm MDS, bạn là một chatbot chỉ trả lời dựa trên tài liệu được cung cấp. "
    "Tuyệt đối không sử dụng kiến thức nội tại, suy luận "
    "Hãy ghi nhớ lịch sử trò chuyện và hãy tương tác với lịch sử để trả lời câu hỏi chính xác"
    "Với những câu hỏi logic hãy suy luận thật kỹ và trả lời thật chính xác"
    "Chỉ trả lời cho máy ảo hoặc thông tin được hỏi chính xác dựa trên tài liệu "
    "Không thêm giải thích, hướng dẫn, hoặc bất kỳ nội dung nào ngoài câu trả lời. "
    "Sau mỗi câu trả lời, hãy ghi nguồn tài liệu mà bạn đã sử dụng để trả lời câu hỏi. "
    "\n\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
if st.session_state['retriever']:
    rag_chain = create_retrieval_chain(st.session_state['retriever'], question_answer_chain)
else:
    st.error("Vui lòng chọn thư mục để tải tài liệu.")
    st.stop()

suggestion_prompt = ChatPromptTemplate.from_messages([
    ("system", "Bạn là một trợ lý thông minh. Dựa trên lịch sử trò chuyện và nội dung tài liệu sau, hãy tạo các câu hỏi gợi ý mà người dùng có thể muốn hỏi tiếp theo. Các câu hỏi phải:\n- Liên quan đến nội dung tài liệu và lịch sử trò chuyện.\n- Ngắn gọn, cụ thể và không lặp lại với các câu đã hỏi trước đó.\n- Trả lời dưới dạng danh sách, mỗi câu một dòng.\n Bạn hãy tạo câu tiêu đề là 'Bạn có muốn hỏi gì nữa không ?', sau đó đưa ra gợi ý"),
    ("human", "Lịch sử trò chuyện:\nNgười dùng: {user_question}\nChatbot: {chatbot_response}\n\nNội dung tài liệu liên quan: {context}\n\nHãy tạo 3 câu hỏi gợi ý:"),
])

suggestion_chain = suggestion_prompt | llm

store = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    runnable=rag_chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

st.markdown(
    """
    <style>
    .chat-message {
        display: flex;
        margin-bottom: 15px;
        width: 100%;
    }
    .chat-message.user {
        justify-content: flex-end; /* Người dùng bên trái */
    }
    .chat-message.user .message-content {
        background-color: #E1F0FA;
        border: 1px solid #D3E4F3;
        border-radius: 10px;
        padding: 10px 15px;
        max-width: 70%;
        color: #333;
        font-size: 17px;
        line-height: 1.5;
    }
    .chat-message.assistant {
        justify-content: flex-start; /* Chatbot bên phải */
    }
    .chat-message.assistant .message-content {
        background-color: #FFFFFF;
        border: 1px solid #E8ECEF;
        border-radius: 10px;
        padding: 10px 15px;
        max-width: 70%;
        color: #333;
        font-size: 17px;
        line-height: 1.5;
    }
    .faq-container, .suggestion-container {
        margin-top: 10px;
        display: block; /* Hiển thị dọc, mỗi câu một dòng */
        margin-left: auto; /* Căn phải để đồng bộ với bong bóng chatbot */
        max-width: 70%; /* Giới hạn chiều rộng giống bong bóng */
    }
    .faq-button, .suggestion-button {
        background: none;
        border: none;
        padding: 0;
        color: #1a73e8; /* Màu xanh giống liên kết */
        text-decoration: underline;
        cursor: pointer;
        font-size: 14px;
        text-align: left;
        margin-bottom: 5px; /* Khoảng cách giữa các dòng */
    }
    .faq-button:hover, .suggestion-button:hover {
        color: #1557b0; /* Màu khi hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)

if st.session_state['chat_history']:
    first_message = st.session_state['chat_history'][0]
    role = 'user' if first_message['role'] == 'human' else first_message['role']
    st.markdown(
        f"""
        <div class="chat-message {role}">
            <div class="message-content">{first_message['content']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    faq_questions = [
        "Bạn là ai?",
        "Cho tôi địa chỉ IP của máy ảo CSKHTT-Mvas-app1-win.",
        "Quy trình vận hành kỹ thuật gồm những bước nào?"
    ]
    st.markdown('<div class="faq-container">', unsafe_allow_html=True)
    for question in faq_questions:
        if st.button(question, key=f"faq_{question}", help="Nhấp để hỏi", use_container_width=False):
            st.session_state['faq_clicked'] = question
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    for i, message in enumerate(st.session_state['chat_history'][1:], start=1):
        role = 'user' if message['role'] == 'human' else message['role']
        st.markdown(
            f"""
            <div class="chat-message {role}">
                <div class="message-content">{message['content']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if role == 'assistant' and i in st.session_state['suggested_questions']:
            suggested_questions = st.session_state['suggested_questions'][i]
            if suggested_questions:
                st.markdown('<div class="suggestion-container">', unsafe_allow_html=True)
                for suggestion in suggested_questions:
                    if st.button(suggestion, key=f"suggestion_{i}_{suggestion}", help="Nhấp để hỏi", use_container_width=False):
                        st.session_state['faq_clicked'] = suggestion
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

user_input = None
if st.session_state['faq_clicked']:
    user_input = st.session_state['faq_clicked']
    st.session_state['faq_clicked'] = None  
else:
    user_input = st.chat_input("Bạn muốn hỏi gì?")

if user_input and st.session_state['selected_folder']:
    st.session_state['chat_history'].append({"role": "human", "content": user_input})

    query = user_input

    try:
        identity_questions = ["bạn là ai", "who are you", "bạn là gì", "what are you"]
        is_identity_question = any(q in user_input.lower() for q in identity_questions)
        
        if is_identity_question:
            chatbot_response = (
                "Tôi là AINoiBo, một mô hình ngôn ngữ lớn được tạo ra bởi Chuyên viên Vũ Phát Đạt - "
                "Phòng Kỹ thuật khai thác - Trung tâm MDS. Tôi là một chatbot, trả lời các câu hỏi "
                "dựa trên tài liệu được cung cấp."
            )
            has_relevant_info = False  
            retrieved_docs = []
        else:
            retrieved_docs = st.session_state['retriever'].invoke(query)
            
            has_relevant_info = False 
            chatbot_response = "Thông tin yêu cầu không có trong tài liệu."
            
            if retrieved_docs:
                with st.spinner("Đang xử lý..."):
                    try:
                        ai_response = conversational_rag_chain.invoke(
                            {
                                "input": user_input,
                                "chat_history": st.session_state['chat_history'],
                                "context": retrieved_docs
                            },
                            {"configurable": {"session_id": st.session_state['session_id']}}
                        )
                        chatbot_response = ai_response["answer"]
                        
                        no_info_phrases = [
                            "không có trong tài liệu", 
                            "không tìm thấy thông tin",
                            "không có thông tin", 
                            "không được đề cập",
                            "không được nhắc đến",
                            "không tìm thấy",
                            "không có dữ liệu",
                            "không liên quan",
                            "không hiểu được nội dung",
                            "không thể hiểu",
                            "không rõ ràng"
                        ]
                        
                        has_relevant_info = not any(phrase in chatbot_response.lower() for phrase in no_info_phrases)
                        
                    except Exception as e:
                        st.error(f"Lỗi khi gọi LLM: {str(e)}")
                        st.stop()
                        
        if has_relevant_info and retrieved_docs:
            sources = list(set(os.path.basename(doc.metadata.get('source', 'Nguồn không xác định')) for doc in retrieved_docs))
            chatbot_response += f"\n\nNguồn: {', '.join(sources)}"
    except Exception as e:
        st.error(f"Lỗi khi truy xuất tài liệu: {str(e)}")
        st.stop()

    st.session_state['chat_history'].append({"role": "assistant", "content": chatbot_response})

    try:
        last_user_question = user_input
        last_chatbot_response = chatbot_response

        context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "Không có nội dung tài liệu."

        suggestion_response = suggestion_chain.invoke({
            "user_question": last_user_question,
            "chatbot_response": last_chatbot_response,
            "context": context
        })
        
        suggested_questions = suggestion_response.content.strip().split('\n')
        suggested_questions = [q.strip() for q in suggested_questions if q.strip()]
        
        suggested_questions = suggested_questions[:3]
        
        message_index = len(st.session_state['chat_history']) - 1
        st.session_state['suggested_questions'][message_index] = suggested_questions
    except Exception as e:
        st.error(f"Lỗi khi sinh câu hỏi gợi ý: {str(e)}")
        suggested_questions = [
            "Bạn có thể hỏi thêm về chủ đề này không?",
            "Có thông tin nào liên quan khác không?",
            "Bạn có thể giải thích chi tiết hơn không?"
        ]
        message_index = len(st.session_state['chat_history']) - 1
        st.session_state['suggested_questions'][message_index] = suggested_questions

    st.rerun()
elif user_input and not st.session_state['selected_folder']:
    st.session_state['chat_history'].append({"role": "human", "content": user_input})
    error_message = "Vui lòng chọn thư mục trước khi hỏi."
    st.session_state['chat_history'].append({"role": "assistant", "content": error_message})
    suggested_questions = [
        "Bạn có thể chọn thư mục dữ liệu không?",
        "Thư mục nào chứa thông tin bạn cần?",
        "Bạn có muốn thử lại với thư mục khác không?"
    ]
    message_index = len(st.session_state['chat_history']) - 1
    st.session_state['suggested_questions'][message_index] = suggested_questions
    st.rerun()