import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _FakeImageAdapter:
    name = "Feishu"

    def __init__(self):
        self.sent_text = []
        self.sent_images = []
        self.sent_documents = []

    async def send(self, chat_id, content, **kwargs):
        self.sent_text.append((chat_id, content, kwargs))
        return SendResult(success=True)

    async def send_image_file(self, chat_id, image_path, **kwargs):
        self.sent_images.append((chat_id, image_path, kwargs))
        return SendResult(success=True)

    async def send_document(self, chat_id, file_path, **kwargs):
        self.sent_documents.append((chat_id, file_path, kwargs))
        return SendResult(success=True)


def _run(coro):
    return asyncio.run(coro)


def _source():
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_chat",
        user_id="ou_user",
        user_name="张三",
        chat_type="dm",
        thread_id="om_parent",
    )


class TestDirectImageEditShortcut(unittest.TestCase):
    def test_image_edit_intent_requires_generation_or_edit_language(self):
        self.assertTrue(GatewayRunner._message_requests_image_edit("帮我改一下这张图"))
        self.assertTrue(GatewayRunner._message_requests_image_edit("没有温柔地看镜头，重改"))
        self.assertTrue(GatewayRunner._message_requests_image_edit("再改一下，改图"))
        self.assertTrue(GatewayRunner._message_requests_image_edit("make this look cinematic"))
        self.assertFalse(GatewayRunner._message_requests_image_edit("你能看到这张图吗"))
        self.assertFalse(GatewayRunner._message_requests_image_edit("这张图里是什么"))

    def test_direct_image_edit_shortcut_generates_and_sends_file(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        adapter = _FakeImageAdapter()
        runner.adapters = {Platform.FEISHU: adapter}

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            input_image = tmp_path / "input.jpg"
            input_image.write_bytes(b"\xff\xd8\xff\x00")
            output_image = tmp_path / "output.png"
            output_image.write_bytes(b"\x89PNG\r\n\x1a\n")

            vision = AsyncMock(
                return_value=json.dumps(
                    {"success": True, "analysis": "地铁车厢内，一位年轻女生低头看手机，竖幅近景。"},
                    ensure_ascii=False,
                )
            )
            captured = {}

            def fake_image_generate(args, **kwargs):
                captured.update(args)
                return json.dumps({"success": True, "image": str(output_image)}, ensure_ascii=False)

            with (
                patch("tools.vision_tools.vision_analyze_tool", vision),
                patch("tools.image_generation_tool._handle_image_generate", fake_image_generate),
            ):
                source = _source()
                event = MessageEvent(
                    source=source,
                    text="帮我改一下，这张图改成女生温柔地看着镜头",
                    message_type=MessageType.TEXT,
                    media_urls=[str(input_image)],
                    media_types=["image/jpeg"],
                    raw_message=SimpleNamespace(),
                )

                result = _run(runner._maybe_handle_direct_image_edit_request(event, source))

        self.assertEqual(result, "")
        self.assertEqual(adapter.sent_images, [("oc_chat", str(output_image), {"metadata": None})])
        self.assertIn("女生温柔地看着镜头", captured["prompt"])
        self.assertIn("地铁车厢内", captured["prompt"])
        self.assertEqual(captured["aspect_ratio"], "landscape")
        self.assertEqual(captured["reference_image_paths"], [str(input_image)])

    def test_direct_image_edit_shortcut_uses_document_fallback(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        adapter = _FakeImageAdapter()
        adapter.send_image_file = AsyncMock(return_value=SendResult(success=False, error="image data error"))
        runner.adapters = {Platform.FEISHU: adapter}

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            input_image = tmp_path / "input.jpg"
            input_image.write_bytes(b"\xff\xd8\xff\x00")
            output_image = tmp_path / "output.png"
            output_image.write_bytes(b"\x89PNG\r\n\x1a\n")

            with (
                patch(
                    "tools.vision_tools.vision_analyze_tool",
                    AsyncMock(
                        return_value=json.dumps(
                            {"success": True, "analysis": "参考图描述"},
                            ensure_ascii=False,
                        )
                    ),
                ),
                patch(
                    "tools.image_generation_tool._handle_image_generate",
                    lambda args, **kwargs: json.dumps(
                        {"success": True, "image": str(output_image)},
                        ensure_ascii=False,
                    ),
                ),
            ):
                source = _source()
                event = MessageEvent(
                    source=source,
                    text="修改成电影感",
                    message_type=MessageType.TEXT,
                    media_urls=[str(input_image)],
                    media_types=["image/jpeg"],
                    raw_message=SimpleNamespace(),
                )

                result = _run(runner._maybe_handle_direct_image_edit_request(event, source))

        self.assertEqual(result, "")
        self.assertEqual(adapter.sent_documents, [("oc_chat", str(output_image), {"metadata": None})])


if __name__ == "__main__":
    unittest.main()
