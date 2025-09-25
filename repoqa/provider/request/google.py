# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import signal
import time

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted

from repoqa.provider.request import construct_message_list


def make_request(
    client: genai.GenerativeModel,
    message: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 1,
    n: int = 1,
    system_msg="You are a helpful assistant good at coding.",
    top_p: float = 1.0,
    top_k: int = -1,
    **kwargs,
) -> genai.types.GenerateContentResponse:
    messages = []
    if system_msg:
        messages.append({"role": "system", "parts": [system_msg]})
    messages.append({"role": "user", "parts": [message]})
    # Build generation config with supported parameters
    gen_config_params = {
        "candidate_count": n,
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }
    
    if top_p != 1.0:
        gen_config_params["top_p"] = top_p
    if top_k != -1:
        gen_config_params["top_k"] = top_k

    return client.generate_content(
        messages,
        generation_config=genai.types.GenerationConfig(**gen_config_params),
        **kwargs,
    )


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def make_auto_request(*args, **kwargs) -> genai.types.GenerateContentResponse:
    ret = None
    while ret is None:
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(100)
            ret = make_request(*args, **kwargs)
            signal.alarm(0)
        except ResourceExhausted as e:
            print("Rate limit exceeded. Waiting...", e.message)
            signal.alarm(0)
            time.sleep(10)
        except GoogleAPICallError as e:
            print(e.message)
            signal.alarm(0)
            time.sleep(1)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            signal.alarm(0)
            time.sleep(1)
    return ret
