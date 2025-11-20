#!/usr/bin/env python3
"""
find_rpi_windows.py

Trouve l'IP d'une Raspberry Pi sur Windows (utilise mDNS, ARP, ping, scan TCP(22), reverse DNS, NetBIOS).
Usage:
  python find_rpi_windows.py            # auto-detect subnet via ipconfig, scan & print candidates
  python find_rpi_windows.py --run-ssh  # propose la commande ssh et la copie dans le presse-papiers (si pyperclip installé)
  python find_rpi_windows.py 192.168.68.0/24  # forcer un /24 à scanner

Notes:
 - Installer 'zeroconf' améliore grandement la détection mDNS: pip install zeroconf
 - Installe 'tabulate' pour un affichage plus joli (pip install tabulate)
 - Installe 'pyperclip' pour copier la commande ssh automatiquement (pip install pyperclip)
"""
import sys
import subprocess
import re
import socket
import ipaddress
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# optional libs
try:
    from zeroconf import Zeroconf, ServiceBrowser, ServiceStateChange
    ZEROCONF_AVAILABLE = True
except Exception:
    ZEROCONF_AVAILABLE = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except Exception:
    HAS_TABULATE = False

try:
    import pyperclip
    HAS_PYPERCLIP = True
except Exception:
    HAS_PYPERCLIP = False

# --- Helpers -----------------------------------------------------------------

def detect_subnet_windows():
    """Detect network by parsing ipconfig output and returning a CIDR (e.g. '192.168.1.0/24')."""
    try:
        out = subprocess.check_output(["ipconfig"], stderr=subprocess.DEVNULL, text=True, encoding="oem", errors="replace")
    except Exception:
        try:
            out = subprocess.check_output(["ipconfig"], stderr=subprocess.DEVNULL, text=True)
        except Exception:
            return None

    # find an IPv4 and subnet mask
    ip = None
    mask = None
    for ln in out.splitlines():
        ln = ln.strip()
        # IPv4 line
        m = re.search(r"IPv4.*?:\s*(\d{1,3}(?:\.\d{1,3}){3})", ln, re.IGNORECASE)
        if m and not ip:
            ip = m.group(1)
        m2 = re.search(r"(Subnet Mask|Masque).*?:\s*(\d{1,3}(?:\.\d{1,3}){3})", ln, re.IGNORECASE)
        if m2 and not mask:
            mask = m2.group(2)
        if ip and mask:
            break
    if not ip or not mask:
        return None
    try:
        net = ipaddress.ip_network(f"{ip}/{mask}", strict=False)
        return str(net)
    except Exception:
        return None

# mDNS (zeroconf) search for 'raspberrypi.local'
def mdns_lookup_raspberry(timeout=2.0):
    if not ZEROCONF_AVAILABLE:
        return []
    found = []
    # Quick approach: try to resolve 'raspberrypi.local' using socket.getaddrinfo first
    try:
        infos = socket.getaddrinfo("raspberrypi.local", None)
        for info in infos:
            addr = info[4][0]
            if addr not in found:
                found.append(addr)
        if found:
            return found
    except Exception:
        pass

    # fallback: scan services using zeroconf (query for hostnames can be heavy; we'll attempt simple approach)
    zc = Zeroconf()
    try:
        # Use a targeted attempt: try to resolve name directly via socket.gethostbyname_ex (may trigger mDNS resolution if Bonjour is installed)
        try:
            r = socket.gethostbyname_ex("raspberrypi.local")
            for ip in r[2]:
                if ip not in found:
                    found.append(ip)
            if found:
                return found
        except Exception:
            pass
        # If above fails, attempt a short wait to allow system mDNS responders (some Windows need Bonjour service)
        time.sleep(0.1)
    finally:
        zc.close()
    return found

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, encoding="oem", errors="replace")
        return out
    except subprocess.CalledProcessError as e:
        return e.output or ""
    except Exception:
        return ""

def parse_arp_table():
    """Return dict ip -> mac from arp -a"""
    mapping = {}
    out = run_cmd(["arp", "-a"])
    for ln in out.splitlines():
        # look for ip and mac
        m = re.search(r"(\d{1,3}(?:\.\d{1,3}){3})\s+([0-9a-fA-F\.\-:]{7,})", ln)
        if m:
            ip = m.group(1)
            mac = m.group(2).strip().replace(".", "-").lower()
            mapping[ip] = mac
    return mapping

def ping(ip, count=1, timeout_ms=500):
    """Return True if ping reply."""
    cmd = ["ping", "-n", str(count), "-w", str(timeout_ms), ip]
    out = run_cmd(cmd)
    return "TTL=" in out.upper()

def fill_arp_for_ip(ip):
    """Ping once to populate arp for a single IP, then parse arp table."""
    ping(ip, count=1, timeout_ms=700)
    arp = parse_arp_table()
    return arp.get(ip, "")

def ssh_banner(ip, port=22, timeout=0.6):
    """Try to connect to port and read banner (non-blocking short). Return banner or ''."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((ip, port))
        # try to read banner (some servers send immediately)
        try:
            data = s.recv(256)
            s.close()
            return data.decode(errors="ignore").strip()
        except Exception:
            s.close()
            return ""
    except Exception:
        return ""

def netbios_name(ip):
    """Try nbtstat -A ip and extract hostname if any."""
    out = run_cmd(["nbtstat", "-A", ip])
    for ln in out.splitlines():
        ln = ln.strip()
        m = re.search(r"^([^\s]+)\s+<00>", ln)
        if m:
            return m.group(1)
    return ""

# fast scan of port 22 across hosts (concurrent)
def scan_port22_for_hosts(subnet, max_workers=150):
    net = ipaddress.ip_network(subnet, strict=False)
    hosts = [str(ip) for ip in net.hosts()]
    alive = []

    def probe(ip):
        b = ssh_banner(ip, port=22, timeout=0.4)
        # if banner non-empty or connect succeeded -> candidate
        if b:
            return (ip, b)
        # else, attempt TCP connect quickly
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.35)
            s.connect((ip, 22))
            s.close()
            return (ip, "")
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(probe, ip): ip for ip in hosts}
        for fut in as_completed(futures):
            res = None
            try:
                res = fut.result()
            except Exception:
                res = None
            if res:
                alive.append(res)
    return alive  # list of (ip, banner)

# --- Main logic --------------------------------------------------------------

def find_candidates(subnet=None, run_full_scan=True):
    candidates = {}  # ip -> dict with evidence
    evidence_order = []

    # 1) try mDNS / raspberrypi.local
    mdns_ips = []
    try:
        mdns_ips = mdns_lookup_raspberry(timeout=2.0)
    except Exception:
        mdns_ips = []
    for ip in mdns_ips:
        candidates.setdefault(ip, {"mdns": True, "arp": "", "banner": "", "dns": "", "nbt": ""})
        evidence_order.append(("mdns", ip))

    # 2) populate arp by pinging subnet broadcast? Instead do targeted quick sweep:
    if not subnet:
        subnet = detect_subnet_windows()
    if not subnet:
        # can't auto-detect, user must pass subnet
        if not subnet:
            print("Impossible de détecter le sous-réseau automatiquement. Lance le script avec un /24 comme argument (ex: 192.168.68.0/24).")
            return {}, subnet

    # quick: fill ARP by doing a short ping sweep of typical host addresses if run_full_scan True.
    if run_full_scan:
        # Prefer scanning port 22 first for speed/precision
        print(f"[+] Scanning {subnet} for SSH (port 22) — rapide et fiable si SSH est activé...")
        ssh_hosts = scan_port22_for_hosts(subnet)
        for ip, banner in ssh_hosts:
            ent = candidates.setdefault(ip, {"mdns": False, "arp": "", "banner": "", "dns": "", "nbt": ""})
            ent["banner"] = banner or ""
            evidence_order.append(("ssh", ip))

        # if no SSH results, attempt a broader ping sweep to populate ARP
        if not ssh_hosts:
            print("[*] Aucun SSH détecté — effectue un ping sweep pour remplir la table ARP (peut prendre quelques secondes)...")
            net = ipaddress.ip_network(subnet, strict=False)
            ips = [str(ip) for ip in net.hosts()]
            def ping_probe(ip):
                return (ip, ping(ip, count=1, timeout_ms=500))
            with ThreadPoolExecutor(max_workers=200) as ex:
                futures = {ex.submit(ping_probe, ip): ip for ip in ips}
                for fut in as_completed(futures):
                    try:
                        ip, ok = fut.result()
                        if ok:
                            candidates.setdefault(ip, {"mdns": False, "arp": "", "banner": "", "dns": "", "nbt": ""})
                            evidence_order.append(("ping", ip))
                    except Exception:
                        pass

    # populate ARP table and annotate candidates
    arp = parse_arp_table()
    for ip in list(candidates.keys()) + list(arp.keys()):
        if ip not in candidates:
            candidates[ip] = {"mdns": False, "arp": "", "banner": "", "dns": "", "nbt": ""}
        if arp.get(ip):
            candidates[ip]["arp"] = arp[ip]

    # for all candidate IPs, attempt reverse DNS and netbios and a short ssh banner if not yet fetched
    ips_to_probe = list(candidates.keys())
    def probe_more(ip):
        out = {}
        out["dns"] = ""
        out["nbt"] = ""
        out["banner"] = candidates[ip].get("banner","")
        try:
            # reverse DNS
            try:
                name, _, _ = socket.gethostbyaddr(ip)
                out["dns"] = name
            except Exception:
                out["dns"] = ""
            # netbios
            try:
                nb = netbios_name(ip)
                out["nbt"] = nb or ""
            except Exception:
                out["nbt"] = ""
            # banner if empty
            if not out["banner"]:
                out["banner"] = ssh_banner(ip, port=22, timeout=0.5)
        except Exception:
            pass
        return (ip, out)

    with ThreadPoolExecutor(max_workers=120) as ex:
        futures = {ex.submit(probe_more, ip): ip for ip in ips_to_probe}
        for fut in as_completed(futures):
            try:
                ip, info = fut.result()
                candidates[ip]["dns"] = info.get("dns","")
                candidates[ip]["nbt"] = info.get("nbt","")
                if info.get("banner"):
                    candidates[ip]["banner"] = info.get("banner")
            except Exception:
                pass

    # Build scored list: heuristics
    scored = []
    for ip, info in candidates.items():
        score = 0
        reasons = []
        if info.get("mdns"):
            score += 50; reasons.append("mDNS")
        dns = info.get("dns","") or ""
        nbt = info.get("nbt","") or ""
        banner = info.get("banner","") or ""
        if "raspberry" in dns.lower() or "raspberry" in nbt.lower():
            score += 30; reasons.append("name contains 'raspberry'")
        if info.get("arp"):
            score += 5; reasons.append("ARP")
        if banner:
            # banner likely means SSH running on Pi — good signal
            score += 20; reasons.append("SSH banner")
            if "raspberry" in banner.lower():
                score += 10; reasons.append("banner contains 'raspberry'")
        # small bonus if ip looks in same /24 as host? omitted for simplicity
        scored.append((score, ip, info, reasons))
    # sort by score desc
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored, subnet

def pretty_print_and_action(scored, subnet, run_ssh=False):
    if not scored:
        print("Aucun hôte détecté.")
        return

    rows = []
    for score, ip, info, reasons in scored:
        rows.append([score, ip, info.get("arp","") or "-", info.get("dns","") or "-", info.get("nbt","") or "-", (info.get("banner","") or "-"), ", ".join(reasons)])
    headers = ["Score","IP","MAC","Reverse DNS","NetBIOS","SSH banner","Why"]
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        # crude print
        print("{:<6} {:<15} {:<18} {:<30} {:<18} {:<20} {:<30}".format(*headers))
        for r in rows:
            print("{:<6} {:<15} {:<18} {:<30} {:<18} {:<20} {:<30}".format(*r))

    best = scored[0]
    score, ip, info, reasons = best
    print("\nMeilleur candidat:")
    print(f"  IP: {ip}")
    print(f"  Score: {score}  (raison: {', '.join(reasons)})")
    if info.get("arp"):
        print(f"  MAC: {info.get('arp')}")
    if info.get("dns"):
        print(f"  Reverse DNS: {info.get('dns')}")
    if info.get("nbt"):
        print(f"  NetBIOS: {info.get('nbt')}")
    if info.get("banner"):
        print(f"  SSH banner: {info.get('banner')}")
    ssh_cmd = f"ssh pi@{ip}"
    print(f"\nCommande SSH suggérée: {ssh_cmd}")

    if HAS_PYPERCLIP:
        try:
            pyperclip.copy(ssh_cmd)
            print("La commande SSH a été copiée dans le presse-papiers.")
        except Exception:
            pass

    if run_ssh:
        print("\nLancement de la commande SSH via la commande système (si 'ssh' est installé sur Windows).")
        try:
            subprocess.run(["ssh", f"pi@{ip}"])
        except Exception as e:
            print("Erreur lors de l'exécution de ssh :", e)
            print("Tu peux lancer manuellement :", ssh_cmd)

# --- Entrée / exécution ------------------------------------------------------

def main():
    run_ssh = False
    subnet = None
    args = [a for a in sys.argv[1:]]
    for a in list(args):
        if a in ("--run-ssh","-r"):
            run_ssh = True
            args.remove(a)
    if args:
        subnet = args[0]
    scored, used_subnet = find_candidates(subnet=subnet, run_full_scan=True)
    if used_subnet is None:
        return
    pretty_print_and_action(scored, used_subnet, run_ssh=run_ssh)

if __name__ == "__main__":
    main()
