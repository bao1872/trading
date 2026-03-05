import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def format_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)

def to_plain(obj: Any, depth: int = 2) -> Any:
    if depth <= 0:
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: to_plain(v, depth - 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v, depth - 1) for v in obj]
    data = getattr(obj, "__dict__", None)
    if isinstance(data, dict) and data:
        return {k: to_plain(v, depth - 1) for k, v in data.items()}
    result: Dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[name] = value
    return result if result else str(obj)

def add_local_quant_paths() -> None:
    p = Path("d:/quant")
    if p.exists():
        sys.path.insert(0, str(p))

def add_qmt_paths(qmt_home: Path) -> None:
    candidates = [
        qmt_home / "bin.x64",
        qmt_home / "bin.x64" / "Lib" / "site-packages",
        qmt_home / "python",
    ]
    if sys.version_info[:2] == (3, 6):
        candidates.insert(1, qmt_home / "bin.x64" / "python36.zip")
    for candidate in candidates:
        if candidate.exists():
            sys.path.insert(0, str(candidate))


def find_userdata(qmt_home: Path) -> Path:
    for name in ("userdata_mini", "userdata"):
        candidate = qmt_home / name
        if candidate.exists():
            return candidate
    for candidate in qmt_home.glob("userdata*"):
        if candidate.is_dir():
            return candidate
    raise RuntimeError(f"未找到userdata目录，请传入--userdata，当前qmt_home={qmt_home}")


class SimpleCallback:
    def on_disconnected(self) -> None:
        print("event", format_json({"type": "disconnected"}))

    def on_stock_asset(self, asset: Any) -> None:
        data = getattr(asset, "__dict__", {"asset": str(asset)})
        print("event", format_json({"type": "asset", "data": data}))

    def on_stock_position(self, position: Any) -> None:
        data = getattr(position, "__dict__", {"position": str(position)})
        print("event", format_json({"type": "position", "data": data}))


def build_account(stock_account_cls: Any, account_id: str, account_type: str) -> Any:
    if account_type:
        return stock_account_cls(account_id, account_type)
    return stock_account_cls(account_id)


def extract_account_infos(infos: Any) -> List[Tuple[str, str, Any]]:
    results: List[Tuple[str, str, Any]] = []
    if infos is None:
        return results
    for info in infos:
        if isinstance(info, dict):
            account_id = info.get("account_id") or info.get("accountID") or info.get("accountId")
            account_type = info.get("account_type") or info.get("accountType")
        else:
            account_id = getattr(info, "account_id", None) or getattr(info, "accountID", None) or getattr(info, "accountId", None)
            account_type = getattr(info, "account_type", None) or getattr(info, "accountType", None)
        if account_id:
            results.append((str(account_id), str(account_type) if account_type else "STOCK", info))
    return results

def load_account_config(path: Path) -> Tuple[Optional[str], Optional[str]]:
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"读取账号配置失败: {path} {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"账号配置格式错误，需为JSON对象: {path}")
    account_id = data.get("account_id") or data.get("accountId") or data.get("accountID")
    account_type = data.get("account_type") or data.get("accountType")
    return (str(account_id) if account_id else None, str(account_type) if account_type else None)

def run_basic_queries(qmt_home: Path, userdata: Path, account_id: Optional[str], account_type: str) -> None:
    add_qmt_paths(qmt_home)
    add_local_quant_paths()
    try:
        from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
        from xtquant.xttype import StockAccount
    except Exception as exc:
        from live.qmt_client import QmtClient
        client = QmtClient(dry_run=True, extra_config={"initial_cash": 1_000_000.0})
        client.connect()
        acct = client.get_account_info()
        print("basic_info", format_json({"account_id": acct.get("account_id", "default"), "account_type": "DRY", "connected": True, "mode": "dry_run"}))
        print("account_infos", format_json([{"account_id": acct.get("account_id", "default"), "account_type": "DRY"}]))
        print("stock_asset", format_json({"cash": acct["cash"], "equity": acct["equity"]}))
        print("stock_positions", format_json(client.get_positions()))
        client.place_order("000001.SZ", "buy", price=10.0, qty=1000, remark="connect_test_buy")
        client.place_order("000001.SZ", "sell", price=10.5, qty=500, remark="connect_test_sell")
        client.place_order("000002.SZ", "sell", price=20.0, qty=100, remark="invalid_sell")
        print("orders", format_json([o.__dict__ for o in client.get_orders()]))
        print("positions_after", format_json(client.get_positions()))
        print("account_after", format_json(client.get_account_info()))
        client.disconnect()
        return

    class _Callback(SimpleCallback, XtQuantTraderCallback):
        pass

    session_id = int(time.time())
    trader = XtQuantTrader(str(userdata), session_id)
    callback = _Callback()
    trader.register_callback(callback)
    trader.start()

    connect_result = trader.connect()
    if connect_result != 0:
        raise RuntimeError(f"连接失败: connect_result={connect_result}")

    infos = trader.query_account_infos()
    info_items = extract_account_infos(infos)
    if not account_id:
        if not info_items:
            raise RuntimeError("未获取到账号列表，请传入--account-id")
        account_id = info_items[0][0]
        if not account_type:
            account_type = info_items[0][1]

    account = build_account(StockAccount, account_id, account_type)
    subscribe_result = trader.subscribe(account)
    if subscribe_result != 0:
        raise RuntimeError(f"订阅失败: subscribe_result={subscribe_result}")

    asset = trader.query_stock_asset(account)
    positions = trader.query_stock_positions(account)

    print("basic_info", format_json({"account_id": account_id, "account_type": account_type, "connected": True}))
    print("account_infos", format_json(to_plain(infos)))
    print("stock_asset", format_json(to_plain(asset)))
    print("stock_positions", format_json(to_plain(positions)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qmt-home", default="D:/国金证券QMT交易端")
    parser.add_argument("--userdata", default="")
    parser.add_argument("--account-id", default="")
    parser.add_argument("--account-type", default="")
    parser.add_argument("--account-config", default="qmt_account_config.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    qmt_home = Path(args.qmt_home)
    userdata = Path(args.userdata) if args.userdata else find_userdata(qmt_home)
    config_path = Path(args.account_config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    cfg_account_id, cfg_account_type = load_account_config(config_path)
    account_id = args.account_id or cfg_account_id
    account_type = args.account_type or cfg_account_type or "STOCK"
    run_basic_queries(qmt_home=qmt_home, userdata=userdata, account_id=account_id, account_type=account_type)
