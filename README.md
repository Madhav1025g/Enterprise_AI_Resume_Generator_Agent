# Enterprise AI Resume Generator Agent

## Overview

Enterprise AI Resume Generator Agent is a FastAPI-based multi-agent system that generates ATS-friendly professional resumes using Generative AI.

The project demonstrates enterprise software engineering principles including:

- Multi-Agent Architecture
- FastAPI REST APIs
- Authentication
- Logging & Monitoring
- ATS Optimization
- Resume Generation
- Resume Review
- Human-Friendly Resume Optimization
- LLM Integration

---

## Architecture

User Request

↓

FastAPI API

↓

Authentication

↓

Orchestrator

- Profile Analyzer Agent
- ATS Optimization Agent
- Resume Writer Agent
- Human Optimizer Agent
- Reviewer Agent

↓

Structured JSON Response

↓

Logs

---

## Technologies

- Python
- FastAPI
- OpenAI GPT-4o-mini
- Pydantic
- REST APIs
- UUID
- Logging

---

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```text
OPENAI_API_KEY=your_openai_api_key
API_KEY_HERE=your_api_key
```

Run the application

```bash
uvicorn main:app --reload
```

Open Swagger UI

```
http://127.0.0.1:8000/docs
```

---

## Features

- Resume Analysis
- ATS Score Generation
- Resume Writing
- Resume Review
- Human-Friendly Resume Optimization
- Enterprise Logging
- API Authentication
- JSON Responses

---

## Project Structure

```
project/
│
├── main.py
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Author

Madhav G