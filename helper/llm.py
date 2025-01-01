import logging

from openai import OpenAI

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def stream_llm(system_prompt, user_prompt, messages_history, tools=None, model="gpt-4o"):
    client = OpenAI()
    params = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            *messages_history,
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "stream": True,
    }

    if tools:
        params["tools"] = tools
        params["parallel_tool_calls"] = True

    completion = client.chat.completions.create(**params)

    tool_calls = []
    for chunk in completion:
        delta = chunk.choices[0].delta

        if delta and delta.content:
            yield delta.content
        elif delta and delta.tool_calls:
            tcchunklist = delta.tool_calls
            for tcchunk in tcchunklist:
                if len(tool_calls) <= tcchunk.index:
                    tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})

                tc = tool_calls[tcchunk.index]
                if tcchunk.id:
                    tc["id"] += tcchunk.id
                if tcchunk.function.name:
                    tc["function"]["name"] += tcchunk.function.name
                if tcchunk.function.arguments:
                    tc["function"]["arguments"] += tcchunk.function.arguments

    yield tool_calls


def structured_output(system_prompt, user_prompt, messages_history, response_format, model="gpt-4o"):
    client = OpenAI()
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                *messages_history,
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            response_format=response_format,
        )

        llm_response = completion.choices[0].message
        if llm_response.parsed:
            return llm_response.parsed
        else:
            raise Exception(llm_response.refusal)
    except Exception as e:
        logger.error("Error in calling LLM or parsing structured output: ", e)
        raise e
