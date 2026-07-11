from __future__ import annotations


def app(environ: object, start_response: object) -> list[bytes]:
    status = "200 OK"
    headers = [("Content-Type", "text/html; charset=utf-8")]
    body = b"""
    <html><body>
    <h1>Redactify</h1>
    <p>This deployment stub is for serverless compatibility. Run the interactive app locally with python app.py.</p>
    </body></html>
    """
    start_response(status, headers)
    return [body]


application = app
