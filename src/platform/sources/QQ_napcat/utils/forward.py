from typing import Any, Dict, List, Optional
import re
import html


def extract_forward_info_from_raw(raw_event: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    从 raw_event['raw'] 中提取转发消息的关键信息。
    
    返回 dict 示例：
    {
        "forward_id": "7575541498967375962",
        "xml_content": "<?xml ...>",
    }
    若未找到，返回 None。
    """
    raw = raw_event["raw"]
    try:
        # Step 1: 获取 elements 列表
        elements = raw.get("elements", [])
        if not isinstance(elements, list):
            return None

        # Step 2: 遍历 elements，找 multiForwardMsgElement
        for elem in elements:
            multi_forward = elem.get("multiForwardMsgElement")
            xml_content = multi_forward.get("xmlContent")
            xml_content = parse_qq_forward_xml(xml_content)

            if xml_content and isinstance(xml_content, str):
                # Step 3: 获取顶层 msgId 作为 forward_id（用于 get_forward_msg）
                forward_id = raw.get("msgId")
                if not forward_id:
                    forward_id = ""
                return {
                    "forward_id": str(forward_id),
                    "forward_preview": xml_content,
                }

    except Exception as e:
        pass


    return None

def parse_qq_forward_xml(xml_content: Optional[str]) -> str:
    """
    专为 QQ 转发消息 XML 设计的解析器。
    
    输入：包含 <msg>...</msg> 的完整 XML 字符串
    输出：多行纯文本，格式如：
        Lord of melina: @冯氏羊肉馆 玛莉卡那么欲求不满...
        Lord of melina: 应该是神妓
        ...
        查看20条转发消息
    
    若解析失败，返回 "[聊天记录]"
    """
    if not xml_content or not isinstance(xml_content, str):
        return "[聊天记录]"

    try:
        # 1. 使用正则提取所有 <title> 和 <summary> 的内部文本
        # 更稳健：避免 XML 解析器对不规范 XML 崩溃（QQ 的 XML 常有转义问题）
        title_texts: List[str] = []
        
        # 匹配 <title ...>内容</title>
        title_matches = re.findall(r'<title\b[^>]*>([^<]*)</title>', xml_content, re.IGNORECASE)
        for t in title_matches:
            clean_t = html.unescape(t.strip())
            if clean_t:
                title_texts.append(clean_t)

        # 匹配 <summary ...>内容</summary>
        summary_match = re.search(r'<summary\b[^>]*>([^<]*)</summary>', xml_content, re.IGNORECASE)
        summary_text = ""
        if summary_match:
            summary_text = html.unescape(summary_match.group(1).strip())

        # 2. 过滤掉首行“群聊的聊天记录”这类标题（通常是第一个 title）
        messages = []
        skip_first = True
        for t in title_texts:
            if skip_first and (t == "群聊的聊天记录" or t.endswith("的聊天记录")):
                skip_first = False
                continue
            messages.append(t)

        # 3. 组装结果
        result_lines = messages
        if summary_text:
            result_lines.append(summary_text)

        if result_lines:
            return "\n".join(result_lines)
        else:
            return "一个[聊天记录]，但是网卡了没加载出来"

    except Exception:
        # 任何异常都 fallback
        return "一个[聊天记录]，但是网卡了没加载出来"