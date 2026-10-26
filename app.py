"""NexusChat AI — Context-Aware Domain Chatbot powered by Groq.

Run with: streamlit run app.py
"""
from __future__ import annotations

import html
import json
import os
import re
import uuid
from datetime import datetime
from typing import Any, Optional

import streamlit as st
from openai import OpenAI

APP_NAME = "NexusChat AI"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ---------------------------------------------------------------------------
# Groq models
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "llama-3.3-70b-versatile"
MODEL_FALLBACKS = (
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct",
)

MAX_HISTORY_MESSAGES = 20
MAX_MESSAGE_CHARS = 4000
MAX_CHATS = 30

# ---------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------
DOMAINS: dict[str, dict[str, Any]] = {
    "education": {
        "label": "📚 Education",
        "tagline": "Tutor, explainer, and study companion",
        "color": "#7c3aed",
        "system": (
            "You are a patient, expert educational tutor. Your goal is to help the user "
            "understand concepts deeply, not just get answers. Always:\n"
            "- Break down complex ideas into simple steps\n"
            "- Use concrete examples and analogies\n"
            "- Adapt explanations to the user's apparent level\n"
            "- Encourage critical thinking with guiding questions\n"
            "- For math/science, show the reasoning, not just the final answer\n"
            "- End complex explanations with a brief summary or check-for-understanding question\n"
            "You cover all academic subjects: math, science, programming, humanities, languages."
        ),
        "suggestions": [
            "Explain machine learning in simple terms with an analogy.",
            "Help me understand the difference between HTTP and HTTPS.",
            "Teach me the basics of Python loops with examples.",
            "What's the best way to study for a math exam?",
            "Explain photosynthesis step by step.",
            "How does a neural network learn? Break it down.",
        ],
    },
    "cybersecurity": {
        "label": "🛡️ Cybersecurity",
        "tagline": "Defensive security expert and ethical advisor",
        "color": "#dc2626",
        "system": (
            "You are a cybersecurity expert focused on DEFENSIVE security, ethical practices, "
            "and education. You help with:\n"
            "- Understanding vulnerabilities (OWASP Top 10, CVEs, attack surfaces)\n"
            "- Best practices for secure coding, network hardening, password hygiene\n"
            "- Threat modeling and risk assessment\n"
            "- Incident response and forensics concepts\n"
            "- Certifications (Security+, CEH, OSCP) and career paths\n"
            "- Compliance frameworks (ISO 27001, NIST, GDPR)\n\n"
            "IMPORTANT: If the user asks for help with actual hacking, exploitation, or "
            "unauthorized access, politely decline and offer to explain the concept "
            "defensively instead (e.g., 'how to protect against XSS' rather than 'how to "
            "inject XSS'). You may discuss theory, tools used by defenders, and lab "
            "environments (TryHackMe, HackTheBox) freely."
        ),
        "suggestions": [
            "Explain the OWASP Top 10 in simple terms.",
            "How do I secure a web application against SQL injection?",
            "What's the difference between symmetric and asymmetric encryption?",
            "Explain zero-trust architecture with a real-world example.",
            "How should I respond to a suspected data breach?",
            "What certifications should I pursue for a cybersecurity career?",
        ],
    },
    "healthcare": {
        "label": "🏥 Healthcare",
        "tagline": "Health information assistant (not medical advice)",
        "color": "#0891b2",
        "system": (
            "You are a knowledgeable healthcare information assistant. You provide "
            "educational information about health, medicine, wellness, and medical concepts.\n\n"
            "CRITICAL DISCLAIMER: You are NOT a doctor and do NOT provide medical advice, "
            "diagnosis, or treatment. Always encourage users to consult licensed healthcare "
            "professionals for personal medical concerns. If a user describes an emergency "
            "(chest pain, difficulty breathing, severe bleeding, suicidal thoughts), "
            "IMMEDIATELY advise them to call emergency services.\n\n"
            "You can help with:\n"
            "- Explaining medical conditions, symptoms, and treatments (general info only)\n"
            "- Nutrition, fitness, and lifestyle information\n"
            "- Understanding medical terminology and lab results\n"
            "- Public health topics (vaccines, epidemiology, prevention)\n"
            "- Mental health information and coping strategies\n"
            "- Navigating the healthcare system\n\n"
            "Always cite general sources of info (e.g., 'per WHO guidelines') when relevant. "
            "Never diagnose or prescribe."
        ),
        "suggestions": [
            "What is the difference between Type 1 and Type 2 diabetes?",
            "Explain how vaccines work in simple terms.",
            "What are evidence-based ways to improve sleep quality?",
            "How do antibiotics work and why is resistance a problem?",
            "What should I know about managing hypertension?",
            "Explain what an MRI scan does and when it's used.",
        ],
    },
    "travel": {
        "label": "✈️ Travel",
        "tagline": "World-travel guide and trip planner",
        "color": "#ea580c",
        "system": (
            "You are an enthusiastic, well-traveled guide and trip-planning assistant. "
            "You help users:\n"
            "- Plan itineraries (day-by-day if asked)\n"
            "- Discover destinations, hidden gems, and local experiences\n"
            "- Understand cultural norms, etiquette, and safety\n"
            "- Navigate logistics: visas, transport, money, SIM cards\n"
            "- Find food recommendations (street food, fine dining, dietary needs)\n"
            "- Budget travel, luxury travel, solo travel, family trips\n"
            "- Handle travel challenges: jet lag, language barriers, lost luggage\n\n"
            "Be specific — mention actual places, neighborhoods, restaurants, transit "
            "options. If the user gives a budget or travel style, tailor to it. "
            "If unsure about current facts (visa rules, prices), say so and suggest "
            "verifying with official sources."
        ),
        "suggestions": [
            "Plan a 5-day trip to Istanbul on a medium budget.",
            "What are the top things to do in Tokyo for first-timers?",
            "Best time of year to visit Northern Pakistan?",
            "How do I travel Europe on $50/day?",
            "Visa requirements for a Pakistani citizen visiting Turkey?",
            "Hidden gems in Dubai beyond the usual tourist spots.",
        ],
    },
    "general": {
        "label": "💬 General Assistant",
        "tagline": "A helpful, versatile assistant",
        "color": "#0d9488",
        "system": (
            "You are NexusChat, a helpful, friendly, and honest general-purpose AI assistant. "
            "You help with anything the user asks — questions, brainstorming, writing, "
            "analysis, coding, planning, and more.\n\n"
            "Always:\n"
            "- Be direct and clear\n"
            "- Admit when you don't know something\n"
            "- Ask clarifying questions when the request is ambiguous\n"
            "- Offer to go deeper if the user wants\n"
            "- Match the user's tone and language"
        ),
        "suggestions": [
            "Help me brainstorm blog post ideas about productivity.",
            "Write a professional email asking for a deadline extension.",
            "Explain quantum computing like I'm 10 years old.",
            "What are some creative ways to learn a new language?",
            "Give me a 30-minute workout for beginners.",
            "How do I start a small business in Pakistan?",
        ],
    },
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_NAME,
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# PREMIUM CSS — Teal/emerald theme (distinct from other projects)
# ===========================================================================
PREMIUM_CSS = """
<style>
:root {
  --nc-ink: #0f172a;
  --nc-ink-2: #475569;
  --nc-ink-3: #94a3b8;
  --nc-bg: #f8fafc;
  --nc-surface: #ffffff;
  --nc-surface-2: #f1f5f9;
  --nc-border: #e2e8f0;
  --nc-border-2: #cbd5e1;
  --nc-primary: #0d9488;
  --nc-primary-2: #0f766e;
  --nc-primary-3: #5eead4;
  --nc-soft: #f0fdfa;
  --nc-soft-2: #ccfbf1;
  --nc-accent: #f59e0b;
  --nc-accent-2: #d97706;
  --nc-success: #10b981;
  --nc-danger: #dc2626;
  --nc-shadow-sm: 0 1px 2px rgba(15,23,42,.05);
  --nc-shadow-md: 0 4px 14px rgba(15,23,42,.08);
  --nc-shadow-lg: 0 12px 36px rgba(15,23,42,.10);
  --nc-radius-sm: 8px;
  --nc-radius-md: 12px;
  --nc-radius-lg: 18px;
  --nc-radius-xl: 26px;
}

[data-testid="stAppViewContainer"] { background: var(--nc-bg); }
[data-testid="stHeader"] {
  background: rgba(248,250,252,.8);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--nc-border);
}
.block-container {
  padding: 1.5rem 2.5rem 4rem;
  max-width: 1500px;
}

/* ---------- Universal readability ---------- */
html, body, .stApp, [data-testid="stAppViewContainer"] {
  color: var(--nc-ink);
}
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"],
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] * {
  color: var(--nc-ink);
}
[data-testid="stCaptionContainer"] *,
[data-testid="stAppViewContainer"] small {
  color: var(--nc-ink-2) !important;
}

/* ---------- Inputs ---------- */
.stTextInput input, .stTextArea textarea, .stNumberInput input,
.stChatInput textarea, [data-testid="stChatInput"] textarea,
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea,
[data-baseweb="select"] > div, .stSelectbox div[data-baseweb="select"] > div {
  background-color: #ffffff !important;
  color: var(--nc-ink) !important;
  -webkit-text-fill-color: var(--nc-ink) !important;
  caret-color: var(--nc-primary) !important;
  border: 1px solid var(--nc-border-2) !important;
  border-radius: var(--nc-radius-sm) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus,
[data-testid="stChatInput"] textarea:focus {
  border-color: var(--nc-primary) !important;
  box-shadow: 0 0 0 3px var(--nc-soft-2) !important;
}

/* Dropdown popovers */
[data-baseweb="popover"], [role="listbox"], [role="option"] {
  background-color: #ffffff !important;
  color: var(--nc-ink) !important;
}
[role="option"]:hover, [aria-selected="true"] {
  background-color: var(--nc-soft) !important;
  color: var(--nc-primary) !important;
}

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
  background: #ffffff !important;
  color: var(--nc-ink) !important;
  -webkit-text-fill-color: var(--nc-ink) !important;
  border: 1px solid var(--nc-border-2) !important;
  border-radius: var(--nc-radius-sm) !important;
  font-weight: 600 !important;
  padding: 0.5rem 1rem !important;
  transition: all .15s ease !important;
  box-shadow: var(--nc-shadow-sm) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--nc-primary) !important;
  color: var(--nc-primary) !important;
  -webkit-text-fill-color: var(--nc-primary) !important;
  transform: translateY(-1px) !important;
  box-shadow: var(--nc-shadow-md) !important;
}
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
  background: var(--nc-primary) !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  border-color: var(--nc-primary) !important;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
  background: var(--nc-primary-2) !important;
  border-color: var(--nc-primary-2) !important;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #042f2e 0%, #134e4a 100%) !important;
  border-right: 1px solid rgba(255,255,255,.08) !important;
}
section[data-testid="stSidebar"] * {
  color: #d1fae5 !important;
  -webkit-text-fill-color: #d1fae5 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] strong {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  font-weight: 700 !important;
}
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
  color: #6ee7b7 !important;
  -webkit-text-fill-color: #6ee7b7 !important;
}
section[data-testid="stSidebar"] hr {
  border-color: rgba(255,255,255,.12) !important;
  margin: 1rem 0 !important;
}
section[data-testid="stSidebar"] .stButton > button {
  background: rgba(255,255,255,.08) !important;
  color: #d1fae5 !important;
  -webkit-text-fill-color: #d1fae5 !important;
  border: 1px solid rgba(255,255,255,.15) !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(255,255,255,.16) !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  border-color: var(--nc-primary-3) !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: var(--nc-primary) !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  border-color: var(--nc-primary) !important;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: rgba(255,255,255,.08) !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  border: 1px solid rgba(255,255,255,.15) !important;
}
section[data-testid="stSidebar"] [data-baseweb="popover"],
section[data-testid="stSidebar"] [role="option"] {
  background: #134e4a !important;
  color: #d1fae5 !important;
  -webkit-text-fill-color: #d1fae5 !important;
}
section[data-testid="stSidebar"] [role="slider"] {
  background: var(--nc-primary-3) !important;
}

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
  background: var(--nc-surface) !important;
  border: 1px solid var(--nc-border) !important;
  border-radius: var(--nc-radius-md) !important;
  padding: 1rem 1.15rem !important;
  box-shadow: var(--nc-shadow-sm) !important;
}
[data-testid="stMetricLabel"] * {
  color: var(--nc-ink-2) !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
  font-size: 0.72rem !important;
}
[data-testid="stMetricValue"] * {
  color: var(--nc-ink) !important;
  font-weight: 750 !important;
  font-size: 1.5rem !important;
}

/* ---------- Expander ---------- */
[data-testid="stExpander"] {
  border: 1px solid var(--nc-border) !important;
  border-radius: var(--nc-radius-md) !important;
  background: var(--nc-surface) !important;
  box-shadow: var(--nc-shadow-sm) !important;
  overflow: hidden !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
  color: var(--nc-ink) !important;
  -webkit-text-fill-color: var(--nc-ink) !important;
  font-weight: 600 !important;
}

/* ---------- Alerts ---------- */
[data-testid="stAlert"] {
  background: var(--nc-surface) !important;
  border-radius: var(--nc-radius-md) !important;
  border: 1px solid var(--nc-border) !important;
}
[data-testid="stAlert"] * { color: var(--nc-ink) !important; }

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {
  border-radius: var(--nc-radius-lg) !important;
  padding: 1rem 1.15rem !important;
  margin-bottom: 0.85rem !important;
  border: 1px solid var(--nc-border) !important;
  background: var(--nc-surface) !important;
  box-shadow: var(--nc-shadow-sm) !important;
}
[data-testid="stChatMessage"] * { color: var(--nc-ink) !important; }

/* User messages — subtle distinct background */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  background: linear-gradient(135deg, #ecfdf5 0%, #f0fdfa 100%) !important;
  border-color: #a7f3d0 !important;
}

/* Assistant messages */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
  background: #ffffff !important;
  border-left: 3px solid var(--nc-primary) !important;
}

/* Chat input */
[data-testid="stChatInput"] {
  background: #ffffff !important;
  border-radius: var(--nc-radius-md) !important;
  border: 1px solid var(--nc-border-2) !important;
}

/* ---------- Code ---------- */
pre, code, [data-testid="stCode"] * {
  background: var(--nc-surface-2) !important;
  color: var(--nc-ink) !important;
  border-radius: var(--nc-radius-sm) !important;
}

/* ---------- Custom hero ---------- */
.nc-hero {
  position: relative;
  overflow: hidden;
  padding: 2.25rem 2.5rem;
  margin-bottom: 1.5rem;
  border-radius: var(--nc-radius-xl);
  background:
    radial-gradient(circle at top right, rgba(13,148,136,.15), transparent 50%),
    radial-gradient(circle at bottom left, rgba(245,158,11,.10), transparent 50%),
    linear-gradient(135deg, #ffffff 0%, #f0fdfa 100%);
  border: 1px solid var(--nc-border);
  box-shadow: var(--nc-shadow-lg);
}
.nc-hero::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 4px;
  background: linear-gradient(90deg, #0d9488, #5eead4, #f59e0b);
}
.nc-kicker {
  display: inline-block;
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--nc-primary) !important;
  -webkit-text-fill-color: var(--nc-primary) !important;
  background: var(--nc-soft-2);
  padding: 0.4rem 0.85rem;
  border-radius: 999px;
  border: 1px solid #99f6e4;
  margin-bottom: 1rem;
}
.nc-title {
  font-size: clamp(1.9rem, 3.4vw, 2.75rem);
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.1;
  margin: 0 0 0.65rem;
  color: var(--nc-ink) !important;
  -webkit-text-fill-color: var(--nc-ink) !important;
}
.nc-title .accent {
  background: linear-gradient(90deg, #0d9488 0%, #14b8a6 50%, #f59e0b 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent !important;
  color: transparent !important;
}
.nc-subtitle {
  font-size: 1.05rem;
  line-height: 1.65;
  color: var(--nc-ink-2) !important;
  -webkit-text-fill-color: var(--nc-ink-2) !important;
  max-width: 800px;
  margin: 0;
}

/* ---------- Domain cards ---------- */
.nc-domain-card {
  padding: 1.35rem 1.4rem;
  border-radius: var(--nc-radius-lg);
  background: var(--nc-surface);
  border: 1px solid var(--nc-border);
  box-shadow: var(--nc-shadow-sm);
  margin-bottom: 0.75rem;
  cursor: pointer;
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
.nc-domain-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--nc-shadow-md);
  border-color: var(--nc-primary);
}
.nc-domain-card .dc-head {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  margin-bottom: 0.55rem;
}
.nc-domain-card .dc-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 42px; height: 42px;
  border-radius: 12px;
  font-size: 1.35rem;
  background: var(--nc-soft);
  border: 1px solid var(--nc-soft-2);
}
.nc-domain-card .dc-name {
  font-size: 1.05rem;
  font-weight: 750;
  color: var(--nc-ink) !important;
  -webkit-text-fill-color: var(--nc-ink) !important;
  margin: 0;
}
.nc-domain-card .dc-tag {
  font-size: 0.82rem;
  color: var(--nc-ink-2) !important;
  -webkit-text-fill-color: var(--nc-ink-2) !important;
  margin: 0;
  line-height: 1.5;
}

/* ---------- Suggestion chips ---------- */
.nc-chip-wrap {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 0.6rem;
}
.nc-chip {
  background: #ffffff;
  border: 1px solid var(--nc-border);
  border-radius: var(--nc-radius-md);
  padding: 0.75rem 1rem;
  font-size: 0.9rem;
  font-weight: 500;
  color: var(--nc-ink) !important;
  -webkit-text-fill-color: var(--nc-ink) !important;
  cursor: pointer;
  transition: all .15s ease;
  box-shadow: var(--nc-shadow-sm);
  text-align: left;
}
.nc-chip:hover {
  border-color: var(--nc-primary);
  background: var(--nc-soft);
  color: var(--nc-primary) !important;
  -webkit-text-fill-color: var(--nc-primary) !important;
}

/* ---------- Chat list item ---------- */
.nc-chat-item {
  padding: 0.65rem 0.85rem;
  border-radius: var(--nc-radius-sm);
  background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.08);
  margin-bottom: 0.4rem;
  font-size: 0.85rem;
  color: #d1fae5 !important;
  -webkit-text-fill-color: #d1fae5 !important;
  line-height: 1.4;
}
.nc-chat-item.active {
  background: rgba(13,148,136,.35);
  border-color: var(--nc-primary-3);
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

/* ---------- Status pill ---------- */
.nc-status {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 700;
}
.nc-status.ok { background: #dcfce7; color: #166534 !important; -webkit-text-fill-color: #166534 !important; }
.nc-status.warn { background: #fef3c7; color: #854d0e !important; -webkit-text-fill-color: #854d0e !important; }
.nc-status.err { background: #fee2e2; color: #991b1b !important; -webkit-text-fill-color: #991b1b !important; }

/* ---------- Info card ---------- */
.nc-info-card {
  padding: 1.35rem 1.4rem;
  border-radius: var(--nc-radius-lg);
  background: var(--nc-surface);
  border: 1px solid var(--nc-border);
  box-shadow: var(--nc-shadow-sm);
  height: 100%;
  transition: transform .2s ease, box-shadow .2s ease;
}
.nc-info-card:hover {
  transform: translateY(-3px);
  box-shadow: var(--nc-shadow-md);
}
.nc-info-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 44px; height: 44px;
  border-radius: 12px;
  background: var(--nc-soft-2);
  color: var(--nc-primary) !important;
  -webkit-text-fill-color: var(--nc-primary) !important;
  font-size: 1.3rem;
  margin-bottom: 0.85rem;
}
.nc-info-card h4 {
  color: var(--nc-ink) !important;
  -webkit-text-fill-color: var(--nc-ink) !important;
  font-size: 1.05rem;
  font-weight: 700;
  margin: 0 0 0.5rem;
}
.nc-info-card p {
  color: var(--nc-ink-2) !important;
  -webkit-text-fill-color: var(--nc-ink-2) !important;
  font-size: 0.92rem;
  line-height: 1.6;
  margin: 0;
}

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

@media (max-width: 768px) {
  .block-container { padding: 1rem 1rem 3rem !important; }
  .nc-hero { padding: 1.5rem 1.25rem; border-radius: var(--nc-radius-lg); }
  .nc-title { font-size: 1.7rem; }
}
</style>
"""
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


# ===========================================================================
# API helpers
# ===========================================================================

def resolve_api_key() -> Optional[str]:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("GROQ_API_KEY", "")).strip() or None
    except Exception:
        return None


def get_groq_client() -> OpenAI:
    key = resolve_api_key()
    if not key:
        raise RuntimeError(
            "Groq API key is missing. Set GROQ_API_KEY in your environment "
            "or in .streamlit/secrets.toml."
        )
    return OpenAI(api_key=key, base_url=GROQ_BASE_URL)


def _classify_groq_error(exc: Exception) -> tuple[str, str]:
    msg = str(exc).lower()
    if any(w in msg for w in ("401", "unauthorized", "invalid api key", "authentication")):
        return ("auth", "Groq authentication failed. Your GROQ_API_KEY is invalid or expired.")
    if any(w in msg for w in ("402", "insufficient", "payment required")):
        return ("billing", "Groq account has insufficient balance or quota.")
    if any(w in msg for w in ("429", "rate limit", "too many requests")):
        return ("rate_limit", "Groq rate limit reached. Please wait a moment and retry.")
    if any(w in msg for w in ("timeout", "deadline", "connection", "network", "unreachable")):
        return ("network", "Could not reach Groq — network or DNS issue.")
    if any(w in msg for w in ("404", "model_not_found", "does not exist",
                              "decommissioned", "no such model", "invalid model")):
        return ("model", "Model unavailable.")
    if any(w in msg for w in ("context length", "too many tokens", "maximum context", "token limit")):
        return ("context", "Conversation is too long for this model. Clear the chat or switch models.")
    if any(w in msg for w in ("400", "bad request")):
        return ("bad_request", f"Groq rejected the request: {str(exc)[:200]}")
    return ("unknown", f"Groq request failed: {str(exc)[:200]}")


def test_groq(model: str) -> tuple[bool, str]:
    if not resolve_api_key():
        return False, "❌ GROQ_API_KEY not set."
    try:
        client = get_groq_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with just: OK"}],
            max_tokens=5,
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return False, f"⚠️ Empty response from `{model}`."
        return True, f"✅ `{model}` is working. Response: {text[:30]}"
    except Exception as exc:
        cat, msg = _classify_groq_error(exc)
        return False, f"❌ [{cat}] {msg}"


def call_groq_chat(
    messages: list[dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1500,
) -> str:
    """Call Groq chat with model fallback."""
    client = get_groq_client()
    candidates = [model] + [m for m in MODEL_FALLBACKS if m != model]

    last_error_msg = "No models tried."
    for model_name in candidates:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            if not resp.choices:
                raise RuntimeError("Groq returned no choices.")
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                raise RuntimeError("Groq returned empty content.")
            try:
                st.session_state["active_model"] = model_name
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    st.session_state["last_usage"] = {
                        "prompt": getattr(usage, "prompt_tokens", 0) or 0,
                        "completion": getattr(usage, "completion_tokens", 0) or 0,
                        "total": getattr(usage, "total_tokens", 0) or 0,
                    }
            except Exception:
                pass
            return content
        except Exception as exc:
            cat, msg = _classify_groq_error(exc)
            last_error_msg = f"[{cat}] {msg}"
            if cat in ("model", "context"):
                continue
            raise RuntimeError(msg)

    raise RuntimeError(f"All Groq models failed. Last: {last_error_msg}")


# ===========================================================================
# Session state
# ===========================================================================

def init_state() -> None:
    defaults: dict[str, Any] = {
        "chats": {},                    # {chat_id: {title, domain, messages, created}}
        "active_chat_id": None,
        "domain": "general",
        "preferred_model": DEFAULT_MODEL,
        "temperature": 0.7,
        "max_tokens": 1500,
        "language": "English",
        "show_debug": False,
        "connection_test": None,
        "active_model": None,
        "last_usage": None,
        "last_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def create_new_chat(domain: str = None) -> str:
    chat_id = str(uuid.uuid4())[:8]
    d = domain or st.session_state.get("domain", "general")
    st.session_state.chats[chat_id] = {
        "title": "New conversation",
        "domain": d,
        "messages": [],
        "created": datetime.now().isoformat(),
    }
    st.session_state.active_chat_id = chat_id
    return chat_id


def get_active_chat() -> Optional[dict[str, Any]]:
    cid = st.session_state.active_chat_id
    if cid and cid in st.session_state.chats:
        return st.session_state.chats[cid]
    return None


def auto_title_chat(chat: dict[str, Any]) -> None:
    """Generate a title from the first user message."""
    if chat["title"] != "New conversation":
        return
    for m in chat["messages"]:
        if m["role"] == "user":
            title = m["content"].strip().replace("\n", " ")[:42]
            if len(m["content"]) > 42:
                title += "…"
            chat["title"] = title or "New conversation"
            return


# ===========================================================================
# Chat engine
# ===========================================================================

def build_system_prompt(domain_key: str, language: str) -> str:
    domain = DOMAINS.get(domain_key, DOMAINS["general"])
    base = domain["system"]
    lang_note = ""
    if language == "Roman Urdu":
        lang_note = (
            "\n\nLANGUAGE: The user may write in English, Roman Urdu (Urdu written in "
            "English letters), or Urdu script. Understand all three. Reply in the SAME "
            "language style as the user's last message. If they mix English and Roman "
            "Urdu, mirror that mix naturally."
        )
    else:
        lang_note = (
            "\n\nLANGUAGE: The user may write in English or Roman Urdu. Understand both, "
            "but reply in English unless the user explicitly writes in Roman Urdu and "
            "you sense they prefer it."
        )
    return base + lang_note


def build_messages_for_api(chat: dict[str, Any], language: str) -> list[dict[str, str]]:
    domain_key = chat.get("domain", "general")
    system = build_system_prompt(domain_key, language)
    api_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    history = chat["messages"][-MAX_HISTORY_MESSAGES:]
    for m in history:
        if m["role"] in ("user", "assistant") and m.get("content"):
            api_messages.append({"role": m["role"], "content": m["content"]})
    return api_messages


def send_message(user_text: str) -> tuple[bool, str]:
    """Append user msg, call Groq, append assistant msg. Returns (ok, error_msg)."""
    chat = get_active_chat()
    if not chat:
        return False, "No active chat."

    user_text = user_text.strip()
    if not user_text:
        return False, "Empty message."
    if len(user_text) > MAX_MESSAGE_CHARS:
        return False, f"Message too long ({len(user_text)} chars). Max {MAX_MESSAGE_CHARS}."

    chat["messages"].append({
        "role": "user",
        "content": user_text,
        "time": datetime.now().isoformat(),
    })

    api_messages = build_messages_for_api(chat, st.session_state.language)

    try:
        reply = call_groq_chat(
            messages=api_messages,
            model=st.session_state.preferred_model,
            temperature=float(st.session_state.temperature),
            max_tokens=int(st.session_state.max_tokens),
        )
        chat["messages"].append({
            "role": "assistant",
            "content": reply,
            "time": datetime.now().isoformat(),
        })
        auto_title_chat(chat)
        st.session_state.last_error = None
        return True, ""
    except RuntimeError as exc:
        err = str(exc)
        st.session_state.last_error = err
        # Add an error placeholder message so user sees feedback
        chat["messages"].append({
            "role": "assistant",
            "content": f"⚠️ **Error:** {err}",
            "time": datetime.now().isoformat(),
            "error": True,
        })
        return False, err


# ===========================================================================
# Export
# ===========================================================================

def export_chat_markdown(chat: dict[str, Any]) -> str:
    domain_key = chat.get("domain", "general")
    domain = DOMAINS.get(domain_key, DOMAINS["general"])
    lines = [
        f"# NexusChat AI — Conversation Export", "",
        f"- **Domain:** {domain['label']}",
        f"- **Created:** {chat.get('created', '')[:19].replace('T', ' ')}",
        f"- **Messages:** {len(chat['messages'])}",
        f"- **Exported:** {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        "", "---", "",
    ]
    for m in chat["messages"]:
        role = "🧑 You" if m["role"] == "user" else "🤖 NexusChat"
        lines.append(f"### {role}")
        lines.append("")
        lines.append(m["content"])
        lines.append("")
    return "\n".join(lines)


def export_chat_json(chat: dict[str, Any]) -> str:
    data = {
        "title": chat.get("title", ""),
        "domain": chat.get("domain", ""),
        "created": chat.get("created", ""),
        "exported": datetime.now().isoformat(),
        "messages": [
            {"role": m["role"], "content": m["content"], "time": m.get("time", "")}
            for m in chat["messages"]
        ],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


# ===========================================================================
# UI Helpers
# ===========================================================================

def esc(s: Any) -> str:
    return html.escape(str(s) if s is not None else "")


def render_hero() -> None:
    st.markdown(
        """
        <div class="nc-hero">
          <div class="nc-kicker">Context-Aware Domain Intelligence</div>
          <div class="nc-title">💬 NexusChat <span class="accent">AI</span></div>
          <div class="nc-subtitle">
            A domain-specialized chatbot that remembers context, adapts to your level,
            and gives meaningful answers — powered by Groq's fastest LLMs.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_landing() -> None:
    st.markdown("### Choose a domain to begin")
    st.caption("Each domain has a specialized system prompt and tone. You can switch anytime.")

    cols = st.columns(3)
    domains_list = list(DOMAINS.keys())
    for i, key in enumerate(domains_list):
        d = DOMAINS[key]
        with cols[i % 3]:
            st.markdown(
                f'<div class="nc-domain-card">'
                f'<div class="dc-head">'
                f'<div class="dc-icon">{d["label"].split(" ", 1)[0]}</div>'
                f'<div>'
                f'<p class="dc-name">{esc(d["label"].split(" ", 1)[1])}</p>'
                f'<p class="dc-tag">{esc(d["tagline"])}</p>'
                f'</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Start {d['label'].split(' ',1)[1]}", key=f"start_{key}",
                         use_container_width=True):
                st.session_state.domain = key
                create_new_chat(domain=key)
                st.rerun()

    st.markdown("")
    st.info("💡 **Tip:** After starting a chat, you can switch domains from the sidebar anytime.")


def render_suggestion_chips(chat: dict[str, Any]) -> None:
    domain_key = chat.get("domain", "general")
    suggestions = DOMAINS.get(domain_key, DOMAINS["general"])["suggestions"]
    st.markdown("### 💡 Try asking")
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(s, key=f"sugg_{chat['created']}_{i}", use_container_width=True):
                send_message(s)
                st.rerun()


def render_chat(chat: dict[str, Any]) -> None:
    domain_key = chat.get("domain", "general")
    domain = DOMAINS.get(domain_key, DOMAINS["general"])

    # Header bar
    header_col1, header_col2 = st.columns([5, 1])
    with header_col1:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;margin-bottom:0.5rem;">'
            f'<span style="background:{domain["color"]}1a;color:{domain["color"]};font-weight:700;'
            f'padding:0.35rem 0.85rem;border-radius:999px;font-size:0.82rem;border:1px solid {domain["color"]}33;">'
            f'{esc(domain["label"])}</span>'
            f'<span style="color:#475569;font-size:0.88rem;">{esc(domain["tagline"])}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with header_col2:
        if st.button("🗑️ Clear", use_container_width=True, key=f"clear_{chat['created']}"):
            chat["messages"] = []
            chat["title"] = "New conversation"
            st.rerun()

    # Messages
    if not chat["messages"]:
        render_suggestion_chips(chat)
    else:
        for m in chat["messages"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    # Chat input
    prompt = st.chat_input(
        "Type your message… (English / Roman Urdu)",
        max_chars=MAX_MESSAGE_CHARS,
        key=f"chat_input_{chat['created']}",
    )
    if prompt:
        with st.spinner("NexusChat is thinking…"):
            ok, err = send_message(prompt)
        if not ok:
            st.error(f"Could not get a response: {err}")
        st.rerun()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    init_state()
    render_hero()

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.markdown("### 🎯 Domain")
        domain_labels = {k: v["label"] for k, v in DOMAINS.items()}
        current_domain = st.session_state.get("domain", "general")
        selected_domain_label = st.selectbox(
            "Active domain",
            options=list(domain_labels.values()),
            index=list(domain_labels.keys()).index(current_domain) if current_domain in domain_labels else 0,
            label_visibility="collapsed",
        )
        selected_domain_key = [k for k, v in domain_labels.items() if v == selected_domain_label][0]

        if selected_domain_key != current_domain:
            st.session_state.domain = selected_domain_key
            # Switch domain on active chat
            active = get_active_chat()
            if active:
                active["domain"] = selected_domain_key
                st.rerun()

        st.markdown("---")
        st.markdown("### 💬 Conversations")

        if st.button("➕ New conversation", use_container_width=True, type="primary"):
            create_new_chat()
            st.rerun()

        # List chats
        chats_sorted = sorted(
            st.session_state.chats.items(),
            key=lambda kv: kv[1].get("created", ""),
            reverse=True,
        )[:MAX_CHATS]

        if not chats_sorted:
            st.caption("No conversations yet.")
        else:
            for cid, c in chats_sorted:
                is_active = cid == st.session_state.active_chat_id
                label = c["title"]
                d_short = DOMAINS.get(c.get("domain", "general"), DOMAINS["general"])["label"].split(" ", 1)[0]
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    if st.button(
                        f"{d_short} {label}",
                        key=f"chat_{cid}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                    ):
                        st.session_state.active_chat_id = cid
                        st.session_state.domain = c.get("domain", "general")
                        st.rerun()
                with col_b:
                    if st.button("✕", key=f"del_{cid}", use_container_width=True):
                        del st.session_state.chats[cid]
                        if st.session_state.active_chat_id == cid:
                            remaining = list(st.session_state.chats.keys())
                            st.session_state.active_chat_id = remaining[0] if remaining else None
                        st.rerun()

        st.markdown("---")
        st.markdown("### ⚙️ Model Settings")
        st.selectbox(
            "Model",
            options=list(MODEL_FALLBACKS),
            index=0,
            key="preferred_model",
        )
        st.slider(
            "Temperature",
            min_value=0.0, max_value=1.5, value=0.7, step=0.05,
            key="temperature",
            help="Lower = focused & factual. Higher = creative & varied.",
        )
        st.slider(
            "Max response tokens",
            min_value=200, max_value=4000, value=1500, step=100,
            key="max_tokens",
        )

        st.markdown("### 🌐 Language")
        st.selectbox(
            "Preferred reply language",
            options=["English", "Roman Urdu"],
            index=0,
            key="language",
        )

        st.markdown("---")
        st.markdown("### 🔌 Connection")
        if resolve_api_key():
            st.success("Groq API key detected")
            col_t, col_c = st.columns(2)
            with col_t:
                if st.button("Test", use_container_width=True):
                    with st.spinner("Testing…"):
                        ok, msg = test_groq(st.session_state.get("preferred_model", DEFAULT_MODEL))
                        st.session_state.connection_test = (ok, msg)
            with col_c:
                if st.button("Clear", use_container_width=True):
                    st.session_state.connection_test = None
            if st.session_state.get("connection_test"):
                ok, msg = st.session_state["connection_test"]
                (st.success if ok else st.error)(msg)
            if st.session_state.get("active_model"):
                st.caption(f"Last used: `{st.session_state['active_model']}`")
            usage = st.session_state.get("last_usage")
            if usage:
                st.caption(f"Tokens — prompt: {usage.get('prompt',0)} · completion: {usage.get('completion',0)}")
        else:
            st.warning("Groq API key not configured")
            st.caption("Set `GROQ_API_KEY` in environment or `.streamlit/secrets.toml`.")

        st.markdown("---")
        st.markdown("### 🐛 Debug")
        st.checkbox("Show debug info", key="show_debug")

        if st.session_state.get("last_error") and st.session_state.get("show_debug"):
            with st.expander("⚠️ Last error", expanded=True):
                st.code(st.session_state["last_error"])

    # ---------------- Main content ----------------
    chat = get_active_chat()

    if chat is None:
        render_landing()
        st.stop()

    # Top toolbar
    t1, t2, t3 = st.columns([3, 1, 1])
    with t1:
        st.markdown(f"### {esc(chat['title'])}")
    with t2:
        md = export_chat_markdown(chat)
        st.download_button(
            "📥 Export MD",
            md,
            file_name=f"nexuschat_{chat['created'][:10]}.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=not chat["messages"],
        )
    with t3:
        js = export_chat_json(chat)
        st.download_button(
            "🧾 Export JSON",
            js,
            file_name=f"nexuschat_{chat['created'][:10]}.json",
            mime="application/json",
            use_container_width=True,
            disabled=not chat["messages"],
        )

    render_chat(chat)

    # Footer
    st.markdown("---")
    st.caption(
        "🔒 **Privacy** — Messages are sent to Groq for AI responses. "
        "Conversations are kept in memory for this session only and are not stored on disk."
    )


if __name__ == "__main__":
    main()
