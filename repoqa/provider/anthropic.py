# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List

from anthropic import Client

from repoqa.provider.base import BaseProvider
from repoqa.provider.request.anthropic import make_auto_request


class AnthropicProvider(BaseProvider):
    def __init__(self, model):
        self.model = model
        self.client = Client(api_key=os.getenv("ANTHROPIC_KEY"))

    def generate_reply(
        self, question, n=1, max_tokens=1024, temperature=0.0,
        top_p=1.0, top_k=-1, repetition_penalty=1.0, 
        presence_penalty=0.0, frequency_penalty=0.0,
        system_msg=None
    ) -> List[str]:
        assert temperature != 0 or n == 1, "n must be 1 when temperature is 0"
        replies = []
        # Anthropic-specific parameter mapping
        kwargs = {}
        if top_p != 1.0:
            kwargs['top_p'] = top_p
        if top_k != -1:
            kwargs['top_k'] = top_k
        # Note: Anthropic doesn't support repetition_penalty, presence_penalty, frequency_penalty

        for _ in range(n):
            reply = make_auto_request(
                self.client,
                message=question,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_msg=system_msg,
                **kwargs,
            )
            replies.append(reply.content[0].text)

        return replies
