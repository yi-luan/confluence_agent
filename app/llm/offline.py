"""離線示範 / 測試用的假 LLM（LLM_PROVIDER=fake）。

依照 system prompt 判斷目前在 Agent 的哪個節點，回傳可預期的內容；
不連網、不花錢，讓沒有 API key 的人也能把整條流程（路由、檢索、引用、監控）跑一遍。
回答內容只是把檢索到的第一段原文接上引用編號，不代表真實 LLM 的回答品質。
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent import prompts


class ScriptedChatModel(BaseChatModel):
    """不連網、決定性的 Chat model。

    - 路由：問題含「誰」「聯絡」「分機」→ directory；含「你好」「謝謝」→ chitchat；其他 → knowledge
    - 回答：把第一段參考資料的第一行內容與 [1] 串起來，方便驗證引用格式
    """

    model_name: str = "offline-scripted"
    calls: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _respond(self, system: str, user: str) -> str:
        if system == prompts.CONDENSE_SYSTEM:
            question = user.split("最新問題：")[-1]
            return question.replace("那病假呢", "病假需要提前多久申請")
        if system == prompts.ROUTER_SYSTEM:
            if any(k in user for k in ("誰", "聯絡", "分機", "窗口")):
                return "directory"
            if any(k in user for k in ("你好", "謝謝", "你是誰")):
                return "chitchat"
            return "knowledge"
        if system == prompts.ANSWER_SYSTEM:
            context = user.split("【問題】")[0]
            first = context.split("[1]", 1)[-1].strip().splitlines()[0]
            return f"{first} [1]"
        if system == prompts.DIRECTORY_EXTRACT_SYSTEM:
            for kw in ("VPN", "報帳", "部署", "請假", "資安"):
                if kw in user:
                    return kw
            return user[-4:]
        if system == prompts.DIRECTORY_ANSWER_SYSTEM:
            return "查詢結果：" + user.split("【通訊錄查詢結果】")[-1].split("【問題】")[0].strip()
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
