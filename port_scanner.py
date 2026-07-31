#!/usr/bin/env python3
"""Simple multithreaded TCP port scanner for authorized testing on hosts you own or have permission to scan."""

import argparse
import socket
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


def parse_ports(port_str):
    ports = set()
    for part in port_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            ports.update(range(int(start), int(end) + 1))
        else:
            ports.add(int(part))
    return sorted(p for p in ports if 0 < p < 65536)


def scan_port(host, port, timeout):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        if result == 0:
            try:
                service = socket.getservbyport(port, "tcp")
            except OSError:
                service = "unknown"
            return port, service
        return None


def main():
    parser = argparse.ArgumentParser(description="Multithreaded TCP port scanner")
    parser.add_argument("target", help="Hostname or IP address to scan")
    parser.add_argument(
        "-p", "--ports", default="1-1024",
        help="Ports to scan, e.g. '22,80,443' or '1-1024' (default: 1-1024)"
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=200,
        help="Number of concurrent threads (default: 200)"
    )
    parser.add_argument(
        "-w", "--timeout", type=float, default=1.0,
        help="Timeout per connection attempt in seconds (default: 1.0)"
    )
    args = parser.parse_args()

    try:
        target_ip = socket.gethostbyname(args.target)
    except socket.gaierror:
        print(f"Error: could not resolve host '{args.target}'")
        sys.exit(1)

    try:
        ports = parse_ports(args.ports)
    except ValueError:
        print(f"Error: invalid port specification '{args.ports}'")
        sys.exit(1)

    print(f"Scanning {args.target} ({target_ip})")
    print(f"Ports: {len(ports)} | Threads: {args.threads} | Timeout: {args.timeout}s")
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    open_ports = []
    start = datetime.now()

    try:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = {
                executor.submit(scan_port, target_ip, port, args.timeout): port
                for port in ports
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    open_ports.append(result)
    except KeyboardInterrupt:
        print("\nScan interrupted by user.")
        sys.exit(1)

    elapsed = (datetime.now() - start).total_seconds()

    print("PORT\tSTATE\tSERVICE")
    for port, service in sorted(open_ports):
        print(f"{port}\topen\t{service}")

    if not open_ports:
        print("No open ports found.")

    print(f"\nScan completed in {elapsed:.2f} seconds. {len(open_ports)} open port(s) found.")


if __name__ == "__main__":
    main()
