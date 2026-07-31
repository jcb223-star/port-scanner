# port-scanner

[![Tests](https://github.com/jcb223-star/port-scanner/actions/workflows/tests.yml/badge.svg)](https://github.com/jcb223-star/port-scanner/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/downloads/)
[![codecov](https://codecov.io/gh/jcb223-star/port-scanner/branch/master/graph/badge.svg)](https://codecov.io/gh/jcb223-star/port-scanner)

A simple multithreaded TCP port scanner, for authorized security testing on hosts you own or have explicit permission to test.

## Usage

```bash
python3 port_scanner.py <target> [-p PORTS] [-t THREADS] [-w TIMEOUT]
```

| Flag | Description | Default |
| --- | --- | --- |
| `-p`, `--ports` | Ports to scan, e.g. `22,80,443` or `1-1024` | `1-1024` |
| `-t`, `--threads` | Number of concurrent threads | `200` |
| `-w`, `--timeout` | Timeout per connection attempt, in seconds | `1.0` |

### Examples

```bash
python3 port_scanner.py 192.168.1.10                       # scan default ports 1-1024
python3 port_scanner.py example.com -p 22,80,443           # scan specific ports
python3 port_scanner.py 10.0.0.5 -p 1-65535 -t 500 -w 0.5  # full scan, more threads, shorter timeout
```

No third-party dependencies — Python 3 standard library only.

## Testing

```bash
python3 -m pytest test_port_scanner.py -v
```
