"""離線示範 / 測試用的假 LLM（LLM_PROVIDER=fake）。

依照 system prompt 判斷目前在 Agent 的哪個節點，回傳可預期的內容；
不連網、不花錢，讓沒有 API key 的人也能把整條流程（路由、檢索、引用、監控）跑一遍。
回答內容只是把檢索到的第一段原文接上引用編號，不代表真實 LLM 的回答品質。
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent import prompts
from app.rag.store import keyword_terms


class ScriptedChatModel(BaseChatModel):
    """不連網、決定性的 Chat model。

    - 路由：問題含「誰」「聯絡」「分機」→ directory；含「你好」「謝謝」→ chitchat；其他 → knowledge
    - 改寫：「那 X 呢？」→ 把上一題的主詞換成 X
    - 回答：從參考資料中挑出和問題關鍵詞重疊最多的一句（抽取式），附上 [編號]
    """

    model_name: str = "offline-scripted"
    calls: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _respond(self, system: str, user: str) -> str:
        if system == prompts.CONDENSE_SYSTEM:
            question = user.split("最新問題：")[-1].strip()
            m = re.fullmatch(r"那(.+?)呢[？?]?", question)
            previous = re.findall(r"user: (.+)", user)
            if m and previous:
                topic, last = m.group(1), previous[-1].strip()
                return topic + last[len(topic) :] if len(last) > len(topic) else topic
            return question
        if system == prompts.ROUTER_SYSTEM:
            if any(k in user for k in ("誰", "聯絡", "分機", "窗口")):
                return "directory"
            if any(k in user for k in ("你好", "謝謝", "你是誰")):
                return "chitchat"
            return "knowledge"
        if system == prompts.ANSWER_SYSTEM:
            context, question = user.split("【問題】")
            terms = set(keyword_terms(question))
            best, best_score = "", -1
            for n, block in re.findall(r"\[(\d+)\] (.*?)(?=\n\n\[\d+\] |\Z)", context, re.S):
                for line in block.splitlines():
                    line = line.strip(" -#*|")
                    if len(line) < 6 or line.startswith("【"):
                        continue
                    score = sum(t in line.lower() for t in terms)
                    if score > best_score:
                        best, best_score = f"{line.replace('**', '')} [{n}]", score
            return best or "知識庫中沒有足夠的資訊回答這個問題"
        if system == prompts.DIRECTORY_EXTRACT_SYSTEM:
            for kw in ("VPN", "報帳", "部署", "請假", "資安"):
                if kw in user:
                    return kw
            return user[-4:]
        if system == prompts.DIRECTORY_ANSWER_SYSTEM:
            rows = user.split("【通訊錄查詢結果】")[-1].split("【問題】")[0].strip()
            return "以下是相關的負責人：\n" + rows
        return "你好！我可以幫你查公司制度與技術文件。"

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        system = messages[0].content if len(messages) > 1 else ""
        user = messages[-1].content
        self.calls.append(system[:20])
        text = self._respond(system, user)
        msg = AIMessage(
            content=text,
            usage_metadata={
                "input_tokens": len(system) + len(user),
                "output_tokens": len(text),
                "total_tokens": len(system) + len(user) + len(text),
            },
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])
