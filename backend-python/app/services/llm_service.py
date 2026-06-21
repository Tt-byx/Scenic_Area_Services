from openai import AsyncOpenAI
import httpx
from app.core.config import settings

_client = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(http_client=httpx.AsyncClient(trust_env=False),
            api_key=settings.mimo_api_key or "not-set",
            base_url=settings.mimo_base_url,
        )
    return _client


SYSTEM_PROMPT = """你是一个专业的景区导游AI数字人助手。你的职责是为游客提供关于景区的准确、友好的信息和建议。

【回答格式要求——必须严格遵守】
1. 禁止使用任何 Markdown 格式符号：不要用 **、*、#、-、1.、```、>、| 等符号。
2. 用纯文本、口语化的方式回答，就像一个真人导游在面对面跟游客聊天。
3. 如果需要列举，用"第一、第二"或"首先、其次"等自然语言连接。
4. 回答要简洁，控制在3-5句话，避免长篇大论。
5. 用中文回答。"""




# ?? ? ??????
EMOTION_STYLE_GUIDE = {
    "happy": "??????????????????????????????",
    "excited": "????????????????????????????",
    "grateful": "????????????????????????????",
    "surprised": "????????????????????????",
    "confused": "???????????????????????????????????",
    "angry": "??????????????????????????????????",
    "worried": "???????????????????????????????",
    "sad": "????????????????????????????????",
}


def _build_system_prompt(user_emotion: str = None) -> str:
    """?????????????"""
    prompt = SYSTEM_PROMPT
    if user_emotion and user_emotion in EMOTION_STYLE_GUIDE:
        prompt += f"\n\n????{EMOTION_STYLE_GUIDE[user_emotion]}"
    return prompt

async def chat_with_mimo(message: str, user_emotion: str = None) -> str:
    """调用 MiMo 大模型 API — 无知识库版本（fallback）"""
    response = await _get_client().chat.completions.create(
        model=settings.mimo_model,
        messages=[
            {"role": "system", "content": _build_system_prompt(user_emotion)},
            {"role": "user", "content": message},
        ],
        temperature=0.7,
        max_tokens=512,
    )
    msg = response.choices[0].message
    return msg.content or ""


async def chat_with_rag(message: str, user_emotion: str = None) -> str:
    """RAG 增强对话：检索知识库 → 拼接上下文 → 调用大模型"""
    from app.services.rag_service import retrieve_context, build_rag_prompt

    context = await retrieve_context(message)
    messages = build_rag_prompt(context, message)

    response = await _get_client().chat.completions.create(
        model=settings.mimo_model,
        messages=messages,
        temperature=0.7,
        max_tokens=512,
    )
    msg = response.choices[0].message
    return msg.content or ""


async def chat_stream(message: str, context_chunks: list[str] = None, user_emotion: str = None):
    """流式对话 — 供 /api/chat/stream 使用"""
    from app.services.rag_service import build_rag_prompt

    if context_chunks:
        context = "\n\n".join(context_chunks)
        messages = build_rag_prompt(context, message)
    else:
        messages = [
            {"role": "system", "content": _build_system_prompt(user_emotion)},
            {"role": "user", "content": message},
        ]

    response = await _get_client().chat.completions.create(
        model=settings.mimo_model,
        messages=messages,
        temperature=0.7,
        max_tokens=512,
        stream=True,
        stop=["\n\n\n"],
    )

    async for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            # 只输出正式回答，跳过思考过程（reasoning_content）
            text = delta.content
            if text:
                yield text
