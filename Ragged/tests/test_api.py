from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from ragged.api.app import create_app


class TestHealthEndpoint:
    @patch("ragged.api.routes._pipeline")
    @patch("ragged.api.routes._config")
    def test_health_returns_status(self, mock_config, mock_pipeline):
        mock_config.ollama.url = "http://localhost:11434"
        mock_pipeline.indexer.list_collections.return_value = []

        app = create_app()
        client = TestClient(app)

        with patch("httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_get.return_value = mock_resp

            resp = client.get("/api/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "status" in data
