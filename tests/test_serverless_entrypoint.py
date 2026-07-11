import importlib
import unittest


class ServerlessEntryPointTests(unittest.TestCase):
    def test_serverless_handler_returns_html(self) -> None:
        module = importlib.import_module("api.index")
        response_headers = {}

        def start_response(status: str, headers: list[tuple[str, str]]) -> None:
            response_headers["status"] = status
            response_headers["headers"] = headers

        body = b"".join(module.app({"PATH_INFO": "/"}, start_response))

        self.assertIn(b"Redactify", body)
        self.assertEqual(response_headers["status"], "200 OK")
        self.assertEqual(response_headers["headers"][0][0], "Content-Type")


if __name__ == "__main__":
    unittest.main()
