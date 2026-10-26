<div align="center">

# 💬 NexusChat AI

### Context-Aware Domain Chatbot

A multi-domain AI chatbot that remembers your conversation and answers in **English or Roman Urdu**, with a specialised assistant for Education, Cybersecurity, Healthcare, Travel and General questions.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama%203.3%2070B-F55036)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

</div>

---

## 📖 Overview

**NexusChat AI** is a web-based chatbot built with Python and Streamlit. The user picks a domain, types a question, and gets an answer from an AI model. Each domain has its own instructions for the AI, so a Healthcare chat behaves differently from a Travel chat. The chatbot remembers earlier messages, so follow-ups such as *"explain that more simply"* work correctly.

> **Internship task:** Develop a context-aware AI chatbot capable of answering user queries related to a specific domain such as education, cybersecurity, healthcare, or travel, maintaining conversation flow and giving meaningful responses using an AI model.
>
> **Level:** Intermediate Level – Project 1 · **Domain:** Artificial Intelligence · **Organisation:** CyberSNova

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **5 specialised domains** | Education, Cybersecurity, Healthcare, Travel and General Assistant |
| 🧠 **Context-aware** | Keeps the last 20 messages so follow-up questions make sense |
| 🌐 **Bilingual replies** | Understands English, Roman Urdu and Urdu script; reply language is selectable |
| 🗂️ **Multiple chats** | Up to 30 chats per session with auto-generated titles |
| 💡 **Suggestion chips** | Ready-made starter questions for each domain |
| 🔁 **Model fallback** | Automatically tries other Groq models if one is unavailable |
| ⚙️ **Tunable settings** | Model, temperature and maximum response length |
| 📤 **Export** | Download any chat as Markdown or JSON |
| 🔌 **Connection test** | Check that your API key works from the sidebar |

### 🎯 Domains

| Domain | Role |
|---|---|
| 📚 **Education** | Patient tutor and study companion across academic subjects |
| 🛡️ **Cybersecurity** | Security explainer and learning guide |
| 🏥 **Healthcare** | General health information assistant |
| ✈️ **Travel** | Trip planning and travel advice |
| 💬 **General Assistant** | Everyday questions on any topic |

> ℹ️ The Healthcare assistant provides general information only and is **not** a substitute for professional medical advice.

---

## 🛠️ Tech Stack

| Tool | Used for |
|---|---|
| **Python 3** | Main programming language |
| **Streamlit** | Web interface: chat window, sidebar, buttons, sliders |
| **Groq API (Llama 3.3 70B)** | AI model that writes the replies |
| **OpenAI Python library** | Client used to call Groq's OpenAI-compatible API |

All AI calls go through one function (`call_groq_chat`), so the model provider can be replaced by changing only that function.

---

## 📁 Project Structure

```text
NexusChat-AI/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .streamlit/
    └── secrets.toml     # (optional, not committed) API key
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- A free [Groq API key](https://console.groq.com/keys)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/NexusChat-AI.git
cd NexusChat-AI
```

### 2. Create a virtual environment *(recommended)*
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

`requirements.txt`:
```text
streamlit
openai
```

### 4. Set your Groq API key

**Option A – Environment variable**
```bash
# Windows (PowerShell)
$env:GROQ_API_KEY="your_api_key_here"

# macOS / Linux
export GROQ_API_KEY="your_api_key_here"
```

**Option B – Streamlit secrets** (`.streamlit/secrets.toml`)
```toml
GROQ_API_KEY = "your_api_key_here"
```

### 5. Run the app
```bash
streamlit run app.py
```
The app opens at **http://localhost:8501**.

---

## 🖥️ Usage

1. Choose a **domain** from the sidebar.
2. Pick the reply language (**English** or **Roman Urdu**).
3. Type a question or click a suggestion chip.
4. Ask follow-up questions; the bot remembers the conversation.
5. Start new chats from the sidebar and export any chat as **Markdown** or **JSON**.

**Sidebar settings**

| Setting | Range | Default |
|---|---|---|
| Temperature | 0.0 – 1.5 | 0.7 |
| Max response tokens | 200 – 4000 | 1500 |
| Reply language | English / Roman Urdu | English |

**Limits:** 20 messages of history sent to the model · 4,000 characters per message · 30 chats per session.

---



## ⚠️ Limitations

- Chats are kept in the browser session only and are lost when the session ends.
- Answers come from an AI model and may contain mistakes; verify important information.
- Free Groq usage is subject to rate limits.

---

## 🔮 Future Improvements

- Permanent chat storage
- More domains and custom domain creation
- Voice input and output

---

## 👤 Author

**Muhammad Numan**
AI Engineering Intern at **[CyberSNova](https://cybersnova.co.in)**


---

<div align="center">

⭐ If you found this project useful, consider giving it a star.

</div>
