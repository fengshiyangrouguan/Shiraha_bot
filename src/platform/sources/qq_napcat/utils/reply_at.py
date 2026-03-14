from typing import Optional, Dict, Any, List


def parse_reply_from_records(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = event.get("raw", {})
    elements = raw.get("elements", [])

    # 1) 找到 replyElement 获取被回复消息的 msgSeq
    reply_msg_seq = None
    for ele in elements:
        r = ele.get("replyElement")
        if r:
            reply_msg_seq = r.get("replayMsgSeq")
            break

    if not reply_msg_seq:
        return None

    records: List[Dict[str, Any]] = raw.get("records", [])
    target = None

    # 2) 找到被回复的消息记录
    for rec in records:
        if str(rec.get("msgSeq")) == str(reply_msg_seq):
            target = rec
            break

    if not target:
        return None

    # 3) 提取原消息文本
    text_parts = []
    for elem in target.get("elements", []):
        te = elem.get("textElement")
        if te and "content" in te:
            text_parts.append(te["content"])

    reply_text = "".join(text_parts).strip() or None

    # 4) 提取原作者昵称（按优先级）
    sender_name = (
        target.get("sendMemberName")
        or target.get("sendNickName")
        or target.get("sendRemarkName")
        or None
    )

    # 兜底：如果昵称为空，用 senderUin 作为识别
    sender_uid = str(target.get("senderUin")) if target.get("senderUin") else None

    return {
        "sender_uid": sender_uid,
        "sender_name": sender_name,
        "text": reply_text,
    }

def get_reply_name(event: Dict[str, Any]) -> str:
    reply_info = parse_reply_from_records({"raw": event})
    if not reply_info:
        return "未知用户"
    return reply_info.get("sender_name") or reply_info.get("sender_uid") or "未知用户"

def get_reply_text(event: Dict[str, Any]) -> str:
    reply_info = parse_reply_from_records({"raw": event})
    if not reply_info:
        return ""
    return reply_info.get("text") or ""


def get_at_nickname(event: dict):
    """只提取 raw.elements 中带 @ 的 textElement.content"""
    contents = []

    elements = event.get("raw", {}).get("elements", [])
    for ele in elements:
        text_ele = ele.get("textElement")
        if not text_ele:
            continue

        content: str = text_ele.get("content", "")
        at_type: int = text_ele.get("atType", 0)

        # 只要是 at，就提取
        if at_type != 0 and content:
            contents.append(content)

    return contents

