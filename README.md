# Hello World - LangChain Starter Project

## Description

A simple introductory project that demonstrates the basics of LangChain. It uses prompt templates and chains to generate summaries and interesting facts about a given person from their biographical information. Great for getting started with LangChain fundamentals.

## Tech Stack

- **LangChain** - LLM framework and chains
- **Google Generative AI (Gemini)** - LLM provider
- **Python 3.12+** - Programming language
- **dotenv** - Environment variable management

## Quick Start

1. Set up environment variables (copy `.env.example` to `.env` and add your Google API key)
2. Install dependencies: `pip install -r requirements.txt` or `uv sync`
3. Run the project: `python main.py`

The script will process sample biographical information and output a summary along with interesting facts using Gemini AI.