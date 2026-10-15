# Live gateway bridge

The existing `thingy-ranging` gateway receives 210-byte stream packets in `Hub.on_result`. This project leaves that repository untouched. To connect the live demo, forward one node's packets to `predict.py live` as NDJSON.

Each stream packet contains `node_id` at byte offset 4, `magic` `0xB5` at offset 5, `count` at offset 7, and companded samples starting at offset 20. An adapter inside the gateway's stream branch can emit:

```python
import json

def classifier_message(packet: bytes) -> str | None:
    if len(packet) != 210 or packet[5] != 0xB5:
        return None
    count = packet[7]
    return json.dumps({
        "node_id": packet[4],
        "codes": list(packet[20:20 + count]),
    }) + "\n"
```

Send the resulting lines to the standard input of `python predict.py live --node-id 0` or a similar subprocess. Keep the gateway's existing streaming mode enabled. Add a dedicated classifier consumer before the current `on_result` early return; the current gateway discards stream chunks when neither noise nor listening is active. This bridge is intentionally separate from model training and remains to be connected and measured on the Raspberry Pi.

Relevant existing code: [`gateway.py` stream dispatch](/home/mixcraftio/Code/Repos/thingy-ranging/thingy_ranging/tools/gateway.py:231), [`conn_time_sync.h` packet layout](/home/mixcraftio/Code/Repos/thingy-ranging/thingy_ranging/include/conn_time_sync.h:161).
