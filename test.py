llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        temperature=0,
        stop=["\nObservation:", "Observation:"],
    )