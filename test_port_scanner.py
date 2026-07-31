import socket
import sys
from unittest.mock import patch

import pytest

import port_scanner


class TestParsePorts:
    def test_single_ports(self):
        assert port_scanner.parse_ports("22,80,443") == [22, 80, 443]

    def test_range(self):
        assert port_scanner.parse_ports("1-5") == [1, 2, 3, 4, 5]

    def test_mixed_ports_and_ranges(self):
        assert port_scanner.parse_ports("22,80,1000-1003") == [22, 80, 1000, 1001, 1002, 1003]

    def test_dedupes_and_sorts(self):
        assert port_scanner.parse_ports("80,22,80,1-3") == [1, 2, 3, 22, 80]

    def test_strips_whitespace(self):
        assert port_scanner.parse_ports(" 22 , 80 ") == [22, 80]

    def test_filters_out_of_range_ports(self):
        assert port_scanner.parse_ports("0,65536,80") == [80]

    def test_invalid_spec_raises(self):
        with pytest.raises(ValueError):
            port_scanner.parse_ports("not-a-port")


class TestScanPort:
    def test_open_port_returns_port_and_service(self):
        with patch.object(socket, "socket") as mock_socket_cls, \
                patch.object(socket, "getservbyport", return_value="http"):
            mock_sock = mock_socket_cls.return_value.__enter__.return_value
            mock_sock.connect_ex.return_value = 0

            assert port_scanner.scan_port("127.0.0.1", 80, 1.0) == (80, "http")

    def test_open_port_with_unknown_service(self):
        with patch.object(socket, "socket") as mock_socket_cls, \
                patch.object(socket, "getservbyport", side_effect=OSError):
            mock_sock = mock_socket_cls.return_value.__enter__.return_value
            mock_sock.connect_ex.return_value = 0

            assert port_scanner.scan_port("127.0.0.1", 31337, 1.0) == (31337, "unknown")

    def test_closed_port_returns_none(self):
        with patch.object(socket, "socket") as mock_socket_cls:
            mock_sock = mock_socket_cls.return_value.__enter__.return_value
            mock_sock.connect_ex.return_value = 1

            assert port_scanner.scan_port("127.0.0.1", 9, 1.0) is None

    def test_sets_timeout(self):
        with patch.object(socket, "socket") as mock_socket_cls:
            mock_sock = mock_socket_cls.return_value.__enter__.return_value
            mock_sock.connect_ex.return_value = 1

            port_scanner.scan_port("127.0.0.1", 9, 2.5)

            mock_sock.settimeout.assert_called_once_with(2.5)


class TestMain:
    def _run_main(self, argv):
        with patch.object(sys, "argv", ["port_scanner.py"] + argv):
            port_scanner.main()

    def test_unresolvable_host_exits_with_error(self, capsys, monkeypatch):
        monkeypatch.setattr(socket, "gethostbyname", lambda host: (_ for _ in ()).throw(socket.gaierror()))

        with pytest.raises(SystemExit) as exc_info:
            self._run_main(["nonexistent.invalid"])

        assert exc_info.value.code == 1
        assert "could not resolve host" in capsys.readouterr().out

    def test_invalid_port_spec_exits_with_error(self, capsys, monkeypatch):
        monkeypatch.setattr(socket, "gethostbyname", lambda host: "127.0.0.1")

        with pytest.raises(SystemExit) as exc_info:
            self._run_main(["127.0.0.1", "-p", "garbage"])

        assert exc_info.value.code == 1
        assert "invalid port specification" in capsys.readouterr().out

    def test_reports_open_ports(self, capsys, monkeypatch):
        monkeypatch.setattr(socket, "gethostbyname", lambda host: "127.0.0.1")
        monkeypatch.setattr(
            port_scanner, "scan_port",
            lambda host, port, timeout: (port, "http") if port == 80 else None,
        )

        self._run_main(["127.0.0.1", "-p", "22,80"])

        out = capsys.readouterr().out
        assert "80\topen\thttp" in out
        assert "22" not in out.split("PORT\tSTATE\tSERVICE")[1].split("\n")[1]

    def test_reports_no_open_ports(self, capsys, monkeypatch):
        monkeypatch.setattr(socket, "gethostbyname", lambda host: "127.0.0.1")
        monkeypatch.setattr(port_scanner, "scan_port", lambda host, port, timeout: None)

        self._run_main(["127.0.0.1", "-p", "22,80"])

        assert "No open ports found." in capsys.readouterr().out
