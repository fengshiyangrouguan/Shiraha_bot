import base64
import logging
import io
from PIL import Image
from typing import Optional

# 导入异步 HTTP 客户端
import aiohttp 
# 导入 aiohttp 的 ClientTimeout，用于设置超时
from aiohttp import ClientTimeout, ContentTypeError

logger = logging.getLogger(__name__)

async def get_image_base64_async(url: str) -> Optional[str]:
    """
    异步获取图片/表情包的Base64。
    使用 aiohttp 确保在事件循环中不会阻塞。
    """
    if not url:
        logger.warning("图片 URL 为空，无法下载。")
        return None
        
    logger.debug(f"异步下载图片: {url}")
    
    # 设置超时：总超时为 10 秒
    timeout = ClientTimeout(total=10) 
    
    # 使用 aiohttp 的 ClientSession 来发起异步请求
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            # 使用 await 发起 GET 请求
            async with session.get(url, ssl=True) as response: 
                
                # 检查 HTTP 状态码
                if response.status != 200:
                    logger.error(f"HTTP 错误: URL={url}, 状态码={response.status}")
                    return None
                
                # 检查 Content-Type，确保它是图片（可选，但推荐）
                content_type = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    logger.warning(f"响应 Content-Type 不是图片: {content_type}")
                    return None

                # 使用 await 读取图片字节
                image_bytes = await response.read() 
                
                # Base64 编码
                return base64.b64encode(image_bytes).decode("utf-8")
        
        # 捕获网络、超时等异常
        except aiohttp.ClientError as e:
            logger.error(f"图片下载失败 (aiohttp 客户端错误): {str(e)}")
            return None
        except Exception as e:
            logger.error(f"图片下载失败 (未知错误): {str(e)}", exc_info=True)
            return None
        

def get_image_format(image_base64: str) -> str:
    """
    从Base64编码的数据中确定图片的格式。
    Parameters:
        raw_data: str: Base64编码的图片数据。
    Returns:
        format: str: 图片的格式（例如 'jpeg', 'png', 'gif'）。
    """
    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes)).format.lower()


def convert_image_to_gif(image_base64: str) -> str:
    # sourcery skip: extract-method
    """
    将Base64编码的图片转换为GIF格式
    Parameters:
        image_base64: str: Base64编码的图片数据
    Returns:
        str: Base64编码的GIF图片数据
    """
    logger.debug("转换图片为GIF格式")
    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        output_buffer = io.BytesIO()
        image.save(output_buffer, format="GIF")
        output_buffer.seek(0)
        return base64.b64encode(output_buffer.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"图片转换为GIF失败: {str(e)}")
        return image_base64